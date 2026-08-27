"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.errorMiddleware = void 0;
const AppError_1 = require("../utils/AppError");
const logger_1 = require("../utils/logger");
const zod_1 = require("zod");
const errorMiddleware = (err, req, res, next) => {
    if (err instanceof zod_1.ZodError) {
        const errors = err.issues.map((e) => ({
            path: e.path.join("."),
            message: e.message,
        }));
        logger_1.logger.warn({ endpoint: req.path, method: req.method, errors }, "[ERROR] Validation error");
        return res.status(400).json({
            success: false,
            message: "Validation Error",
            errors,
        });
    }
    if (err instanceof AppError_1.AppError) {
        logger_1.logger.warn({
            endpoint: req.path,
            method: req.method,
            statusCode: err.statusCode,
            message: err.message,
        }, "[ERROR] Application error");
        return res.status(err.statusCode).json({
            success: false,
            message: err.message,
        });
    }
    // always log at least the message; include stack only when debug
    const baseError = {
        endpoint: req.path,
        method: req.method,
        errorMessage: err.message,
    };
    if (logger_1.logger.isLevelEnabled("debug")) {
        baseError.stack = err.stack;
    }
    logger_1.logger.error(baseError, "[ERROR] Unhandled error");
    return res.status(500).json({
        success: false,
        message: "Internal Server Error",
    });
};
exports.errorMiddleware = errorMiddleware;
