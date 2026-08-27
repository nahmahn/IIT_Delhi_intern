# Middleware Service

This project includes structured logging powered by [pino](https://github.com/pinojs/pino). The behaviour changes depending on the `NODE_ENV` and optional `LOG_LEVEL` environment variables.

## Log Levels

- `NODE_ENV=production`
  - Default level: `info` (stack traces and debug details are omitted).
  - Pretty‑printing disabled by default.
- `NODE_ENV=development` or unset
  - Default level: `debug` (all request/response details, timings and stacks are emitted).
  - Output is formatted with `pino-pretty` for readability.

You can explicitly override with `LOG_LEVEL` (`debug`, `info`, `warn`, `error`, etc.).

Example:

```bash
# development with full debug output
LOG_LEVEL=debug npm run dev

# development with basic info only
LOG_LEVEL=info npm run dev

# production behaviour
NODE_ENV=production npm run start
```

## Request logging

The `requestLoggerMiddleware` logs:

- An initial info message when the request arrives (method, path, URL, client IP).
- A final info message when the response is sent (status code and duration).
- When the logger level is `debug`, it also logs headers, query parameters, route params, request body and response headers.

Errors are handled by `error.middleware.ts` and include stack traces only at debug level.

Feel free to adjust log levels or add additional context in controllers/services as needed.
