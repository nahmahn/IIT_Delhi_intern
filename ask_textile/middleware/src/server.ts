import app from "./app";
import { env } from "./config/env";
import { logger } from "./utils/logger";
import prisma from "./config/prisma";

// show useful diagnostics on startup
const server = app.listen(env.PORT, () => {
  logger.info(
    {
      port: env.PORT,
      nodeEnv: process.env.NODE_ENV,
      logLevel: process.env.LOG_LEVEL || undefined,
    },
    `Server running on port ${env.PORT}`,
  );
});

const shutdown = async () => {
  logger.info("Shutting down server...");
  await prisma.$disconnect();
  server.close();
};

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
