"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.isProduction = exports.isInfo = exports.isDebug = exports.logger = void 0;
const pino_1 = __importDefault(require("pino"));
// Determine log level using explicit LOG_LEVEL or fall back to environment
// NODE_ENV = production -> info (unless overridden)
// otherwise default to debug for development/other environments.
const logLevel = process.env.LOG_LEVEL ||
    (process.env.NODE_ENV === "production" ? "info" : "debug");
exports.logger = (0, pino_1.default)({
    level: logLevel,
    transport: process.env.NODE_ENV !== "production" || logLevel === "debug"
        ? { target: "pino-pretty" }
        : undefined,
});
// Export convenience helpers to make checks easier in other modules
const isDebug = () => exports.logger.isLevelEnabled("debug");
exports.isDebug = isDebug;
const isInfo = () => exports.logger.isLevelEnabled("info");
exports.isInfo = isInfo;
const isProduction = () => process.env.NODE_ENV === "production";
exports.isProduction = isProduction;
