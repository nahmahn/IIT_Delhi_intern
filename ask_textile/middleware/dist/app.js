"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const auth_routes_1 = __importDefault(require("./modules/auth/auth.routes"));
const chat_routes_1 = __importDefault(require("./modules/chat/chat.routes"));
const error_middleware_1 = require("./middlewares/error.middleware");
const cors_1 = require("./config/cors");
const request_logger_middleware_1 = require("./middlewares/request-logger.middleware");
const app = (0, express_1.default)();
app.use(cors_1.corsConfig);
app.use(express_1.default.json());
// Body parsing error handler (catches JSON parsing errors)
app.use(request_logger_middleware_1.bodyParsingErrorHandler);
// Request/Response logging middleware
app.use(request_logger_middleware_1.requestLoggerMiddleware);
app.use("/api/v1/auth", auth_routes_1.default);
app.use("/api/v1/chat", chat_routes_1.default);
//! Global error handler (ALWAYS LAST)
app.use(error_middleware_1.errorMiddleware);
exports.default = app;
