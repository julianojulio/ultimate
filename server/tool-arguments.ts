import type { Context } from '@earendil-works/chord'
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai'
import type {
  Api,
  AssistantMessage,
  AssistantMessageEventStream,
  Model,
  Models,
} from '@earendil-works/pi-ai'
import { defineExtension, GenerationTask, hook, ProviderDoc } from '@earendil-works/pi-durable'
import type { HookApi } from '@earendil-works/pi-durable'
import { traceContext } from './trace.ts'
import type { CallTrace } from './trace.ts'

export const MAX_WHITESPACE_RUN = 1000

export type ModelTrace = {
  begin: CallTrace['begin']
  conversationOf(sessionId: string | undefined): number | undefined
}

export function modelTrace(trace: Pick<CallTrace, 'begin'>) {
  const sessions = new Map<string, number>()
  const remember = async (api: HookApi, context: Context) => {
    const provider = await api.snapshot(ProviderDoc, api.conversationId, context)
    if (!provider) return
    sessions.delete(provider.sessionId)
    sessions.set(provider.sessionId, api.conversationId)
    if (sessions.size > 10_000) sessions.delete(sessions.keys().next().value!)
  }
  return {
    begin: trace.begin,
    conversationOf: (sessionId: string | undefined) =>
      sessionId === undefined ? undefined : sessions.get(sessionId),
    extension: defineExtension({
      name: 'model-trace',
      hooks: [
        hook(GenerationTask, {
          beforeRequest: async (_request, api, context) => {
            await remember(api, context)
          },
        }),
      ],
    }),
  } satisfies ModelTrace & { extension: unknown }
}

const PROVIDER_LIMITS: Record<string, number> = {
  'cirthan-dev': 3,
  cirthan: 3,
}

class ProviderLimiter {
  private active = new Map<string, number>()
  private queues = new Map<string, (() => void)[]>()

  async acquire(provider: string, signal?: AbortSignal): Promise<() => void> {
    const limit = PROVIDER_LIMITS[provider] ?? Infinity
    if (limit === Infinity) return () => {}

    for (;;) {
      const current = this.active.get(provider) ?? 0
      if (current < limit) {
        this.active.set(provider, current + 1)
        let released = false
        return () => {
          if (released) return
          released = true
          const nextCount = (this.active.get(provider) ?? 1) - 1
          if (nextCount <= 0) this.active.delete(provider)
          else this.active.set(provider, nextCount)
          const queue = this.queues.get(provider)
          if (queue && queue.length > 0) {
            const next = queue.shift()!
            if (!queue.length) this.queues.delete(provider)
            next()
          }
        }
      }

      await new Promise<void>((resolve, reject) => {
        if (signal?.aborted) return reject(signal.reason)
        const onAbort = () => {
          const queue = this.queues.get(provider)
          if (queue) {
            const idx = queue.indexOf(resume)
            if (idx >= 0) queue.splice(idx, 1)
            if (!queue.length) this.queues.delete(provider)
          }
          reject(signal?.reason)
        }
        const resume = () => {
          signal?.removeEventListener('abort', onAbort)
          resolve()
        }
        signal?.addEventListener('abort', onAbort, { once: true })
        let queue = this.queues.get(provider)
        if (!queue) {
          queue = []
          this.queues.set(provider, queue)
        }
        queue.push(resume)
      })
    }
  }
}

const limiter = new ProviderLimiter()

export function guardToolArguments<T extends Models>(models: T, trace?: ModelTrace): T {
  const streamSimple: Models['streamSimple'] = (model, context, options) => {
    const stream = guardStream(
      (signal) => {
        const guarded = createAssistantMessageEventStream()
        const run = async () => {
          let release: (() => void) | undefined
          try {
            release = await limiter.acquire(model.provider, signal)
            const inner = models.streamSimple(model, context, { ...options, signal })
            for await (const event of inner) {
              guarded.push(event)
            }
            guarded.end(await inner.result())
          } catch (err: unknown) {
            guarded.push({
              type: 'error',
              reason: 'error',
              error: {
                role: 'assistant',
                content: [],
                api: model.api,
                provider: model.provider,
                model: model.id,
                usage: {
                  input: 0,
                  output: 0,
                  cacheRead: 0,
                  cacheWrite: 0,
                  totalTokens: 0,
                  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
                },
                stopReason: 'error',
                timestamp: Date.now(),
                errorMessage: err instanceof Error ? err.message : String(err),
              },
            })
            guarded.end()
          } finally {
            release?.()
          }
        }
        void run().catch((error: unknown) => {
          // Stream error is already reported on guarded stream
          console.error('model stream failure:', error)
        })
        return guarded
      },
      model,
      options?.signal,
    )
    if (trace) record(trace, stream, model, options?.sessionId)
    return stream
  }
  const completeSimple: Models['completeSimple'] = (model, context, options) =>
    streamSimple(model, context, options).result()
  return new Proxy(models, {
    get(target, key) {
      if (key === 'streamSimple') return streamSimple
      if (key === 'completeSimple') return completeSimple
      const value: unknown = Reflect.get(target, key)
      return typeof value === 'function' ? value.bind(target) : value
    },
  })
}

function record(
  trace: ModelTrace,
  stream: AssistantMessageEventStream,
  model: Pick<Model<Api>, 'provider' | 'id'>,
  sessionId: string | undefined,
) {
  const conversationId = trace.conversationOf(sessionId)
  const { finish } = trace.begin(
    {
      kind: 'model',
      caller: sessionId === undefined ? 'extension' : 'agent',
      callee: model.provider,
      name: `${model.provider}/${model.id}`,
      ...(conversationId === undefined ? {} : { conversationId }),
    },
    sessionId === undefined ? (traceContext() ?? null) : null,
  )
  void stream.result().then(
    (message) => {
      const { input, output, cacheRead, cacheWrite } = message.usage
      const result = { input, output, cacheRead, cacheWrite, stopReason: message.stopReason }
      if (message.stopReason === 'error' || message.stopReason === 'aborted')
        finish(false, new Error(classify(message.stopReason, message.errorMessage)))
      else finish(true, result)
    },
    (error: unknown) =>
      finish(false, new Error(classify('error', error instanceof Error ? error.message : ''))),
  )
}

const CODES = new Set([
  'invalid_grant',
  'invalid_token',
  'invalid_client',
  'invalid_request',
  'invalid_api_key',
  'unauthorized_client',
  'access_denied',
  'authentication_error',
  'permission_error',
  'insufficient_quota',
  'rate_limit_error',
  'rate_limit_exceeded',
  'usage_limit_reached',
  'overloaded_error',
  'api_error',
  'server_error',
  'not_found_error',
  'model_not_found',
  'invalid_request_error',
  'context_length_exceeded',
  'request_too_large',
  'unauthorized',
  'forbidden',
  'expired',
  'revoked',
  'overloaded',
  'timeout',
  'aborted',
  'quota',
  'unavailable',
  'ECONNREFUSED',
  'ECONNRESET',
  'ENOTFOUND',
  'ETIMEDOUT',
  'EAI_AGAIN',
])

export function classify(stopReason: string, message = '') {
  const parts = [stopReason]
  const status = /^\s*([45]\d\d)\b|\b(?:status|http)[\s:=]*([45]\d\d)\b/i.exec(message)
  if (status) parts.push(status[1] ?? status[2])
  for (const word of message.split(/[^A-Za-z_]+/)) {
    const code = CODES.has(word) ? word : CODES.has(word.toLowerCase()) ? word.toLowerCase() : ''
    if (code && !parts.includes(code)) parts.push(code)
    if (parts.length >= 5) break
  }
  return parts.join(' ')
}

export function guardStream(
  open: (signal: AbortSignal) => AssistantMessageEventStream,
  model: Pick<Model<Api>, 'api' | 'provider' | 'id'>,
  signal?: AbortSignal,
): AssistantMessageEventStream {
  const controller = new AbortController()
  const source = open(signal ? AbortSignal.any([signal, controller.signal]) : controller.signal)
  const guarded = createAssistantMessageEventStream()
  const runs = new Map<number, number>()
  let partial: AssistantMessage = {
    role: 'assistant',
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: 'error',
    timestamp: Date.now(),
  }
  const fail = (errorMessage: string) => {
    const error = { ...partial, content: [...partial.content], stopReason: 'error' as const }
    guarded.push({ type: 'error', reason: 'error', error: { ...error, errorMessage } })
  }
  void (async () => {
    for await (const event of source) {
      if ('partial' in event) partial = event.partial
      if (event.type === 'toolcall_delta') {
        let run = runs.get(event.contentIndex) ?? 0
        for (const char of event.delta) {
          run = /\s/.test(char) ? run + 1 : 0
          if (run > MAX_WHITESPACE_RUN) break
        }
        runs.set(event.contentIndex, run)
        if (run > MAX_WHITESPACE_RUN) {
          controller.abort()
          const call = event.partial.content[event.contentIndex]
          const name = call?.type === 'toolCall' ? `${call.name} ` : ''
          const errorMessage = `Aborted degenerate tool-call arguments: ${name}arguments contain more than ${MAX_WHITESPACE_RUN} whitespace characters in a row; you can retry your request`
          fail(errorMessage)
          return
        }
      }
      guarded.push(event)
      if (event.type === 'done' || event.type === 'error') return guarded.end()
    }
    guarded.end(await source.result())
  })().catch((error: unknown) => {
    fail(`Model stream failed: ${String(error)}`)
    guarded.end()
  })
  return guarded
}
