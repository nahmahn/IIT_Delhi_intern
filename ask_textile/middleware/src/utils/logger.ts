import pino from "pino";

// Determine log level using explicit LOG_LEVEL or fall back to environment
// NODE_ENV = production -> info (unless overridden)
// otherwise default to debug for development/other environments.
const logLevel =
  process.env.LOG_LEVEL ||
  (process.env.NODE_ENV === "production" ? "info" : "debug");

export const logger = pino({
  level: logLevel,
  transport:
    process.env.NODE_ENV !== "production" || logLevel === "debug"
      ? { target: "pino-pretty" }
      : undefined,
});

// Export convenience helpers to make checks easier in other modules
export const isDebug = () => logger.isLevelEnabled("debug");
export const isInfo = () => logger.isLevelEnabled("info");
export const isProduction = () => process.env.NODE_ENV === "production";
