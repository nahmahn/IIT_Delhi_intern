# Project Setup Guide

This document describes how to get the middleware service up and running, including
installing dependencies, configuring the database, enabling logging, and testing with
example APIs.

## Prerequisites

- Node.js 18+ (LTS)
- npm or yarn
- A PostgreSQL database (or any database supported by Prisma; connection URL in `DATABASE_URL`)
- Git (optional)

## 1. Clone & install

```bash
git clone https://your-repo-url.git middleware
cd middleware
npm install   # or yarn
```

## 2. Environment configuration

Create a `.env` file in the project root or export environment variables directly. The
`src/config/env.ts` schema validates the following variables:

```text
PORT=3000
DATABASE_URL="postgresql://user:pass@localhost:5432/dbname?schema=public"
JWT_SECRET=some_long_random_string
BCRYPT_COST=12
NODE_ENV=development       # "development" or "production"
LOG_LEVEL=debug            # optional; overrides default log level
```

Example `.env`:

```env
PORT=3000
DATABASE_URL="postgresql://localhost:5432/middleware"
JWT_SECRET=change-me
BCRYPT_COST=12
NODE_ENV=development
LOG_LEVEL=debug
```

> **Note:** `LOG_LEVEL` can be `debug`, `info`, `warn`, `error`. In production the
> default level is `info` (unless overridden).

## 3. Database setup (Prisma)

1. Install Prisma CLI if you haven't already:

   ```bash
   npx prisma --version
   ```

2. Define your schema in `prisma/schema.prisma`. A sample migration already exists under
   `prisma/migrations`.

3. Apply migrations and generate the client:

   ```bash
   npx prisma migrate dev --name init
   ```

   or (during development) just run `npx prisma migrate dev`.

4. If you need to view data:
   ```bash
   npx prisma studio
   ```

## 4. Running the service

- **Development:**

  ```bash
  export NODE_ENV=development
  export LOG_LEVEL=debug      # optional
  npm run dev
  ```

  This uses `ts-node-dev` with live reload. Logs appear prettily formatted.

- **Production build:**
  ```bash
  npm run build
  export NODE_ENV=production
  npm start
  ```

The server listens on `http://localhost:<PORT>` (default 3000).

## 5. Logging behaviour

Logging is powered by [pino](https://github.com/pinojs/pino). The behaviour is:

- **Debug level** (development default)
  - Logs full request/response details (headers, body, params, duration).
  - Errors include stack traces.
  - Output is pretty-printed.

- **Info level** (production default or when `LOG_LEVEL=info`)
  - Only logs request arrival/completion, status, and duration.
  - Debug details are suppressed.
  - No sensitive data (passwords, tokens) ever shown.

Control level with `NODE_ENV` (production vs non-production) or by setting
`LOG_LEVEL` manually.

## 6. Example APIs

The service exposes authentication routes under `/api/v1/auth`.

- **Register**

  ```bash
  curl -X POST http://localhost:3000/api/v1/auth/register \
    -H "Content-Type: application/json" \
    -d '{"displayName":"John","email":"john@example.com","password":"Secret"}'
  ```

- **Login**

  ```bash
  curl -X POST http://localhost:3000/api/v1/auth/login \
    -H "Content-Type: application/json" \
    -d '{"email":"john@example.com","password":"Secret"}'
  ```

  Response contains an `accessToken` and `refreshToken`.

- **Refresh token**

  ```bash
  curl -X POST http://localhost:3000/api/v1/auth/refresh \
    -H "Content-Type: application/json" \
    -d '{"refreshToken":"<token>"}'
  ```

- **Get current user** (protected)

  ```bash
  curl -X GET http://localhost:3000/api/v1/auth/me \
    -H "Authorization: Bearer <accessToken>"
  ```

- **Logout** (protected)
  ```bash
  curl -X POST http://localhost:3000/api/v1/auth/logout \
    -H "Authorization: Bearer <accessToken>" \
    -H "Content-Type: application/json" \
    -d '{"refreshToken":"<token>"}'
  ```

The above commands can be used to verify database connectivity and logging output.

## 7. Troubleshooting

- **Port already in use**: kill existing node process (`npx kill-port 3000`).
- **Migration errors**: inspect `prisma/migrations` or reset with `npx prisma migrate reset`.
- **Missing env vars**: the startup log prints current values for port and log level.

---

This guide should let you set up the project quickly, verify that the database is
working, and understand how logging behaves in different modes.
