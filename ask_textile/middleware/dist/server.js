"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const app_1 = __importDefault(require("./app"));
const env_1 = require("./config/env");
const logger_1 = require("./utils/logger");
const prisma_1 = __importDefault(require("./config/prisma"));
// show useful diagnostics on startup
const server = app_1.default.listen(env_1.env.PORT, () => {
    logger_1.logger.info({
        port: env_1.env.PORT,
        nodeEnv: process.env.NODE_ENV,
        logLevel: process.env.LOG_LEVEL || undefined,
    }, `Server running on port ${env_1.env.PORT}`);
});
const shutdown = () => __awaiter(void 0, void 0, void 0, function* () {
    logger_1.logger.info("Shutting down server...");
    yield prisma_1.default.$disconnect();
    server.close();
});
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
