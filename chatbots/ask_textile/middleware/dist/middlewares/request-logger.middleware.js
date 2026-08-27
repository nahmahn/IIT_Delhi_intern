"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.bodyParsingErrorHandler = exports.requestLoggerMiddleware = void 0;
const logger_1 = require("../utils/logger");
/**
 * Enhanced request logging middleware
 * Logs incoming request details and response information
 */
const requestLoggerMiddleware = (req, res, next) => {
    const startTime = Date.now();
    // Capture original end function
    const originalEnd = res.end;
    // Basic information logged at info level (always shown when level >= info)
    logger_1.logger.info({
        method: req.method,
        path: req.path,
        url: req.originalUrl,
        ip: req.ip,
        status: res.statusCode,
        duration: "pending",
    }, "[REQUEST] Incoming request");
    // When debug logging is enabled show headers, query, params and body
    // helper to mask out sensitive fields when logging body
    const sanitize = (obj) => {
        if (obj && typeof obj === "object") {
            return JSON.parse(JSON.stringify(obj, (key, value) => {
                if (key === "password" ||
                    key === "refreshToken" ||
                    key === "token" ||
                    key === "accessToken") {
                    return "***";
                }
                return value;
            }));
        }
        return obj;
    };
    if (logger_1.logger.isLevelEnabled("debug")) {
        logger_1.logger.debug({
            headers: {
                contentType: req.headers["content-type"],
                authorization: req.headers.authorization ? "present" : "missing",
            },
            query: req.query,
            params: req.params,
        }, "[REQUEST] Expanded request details");
        if (req.method !== "GET" && req.method !== "HEAD") {
            logger_1.logger.debug({
                body: sanitize(req.body),
            }, "[REQUEST] Request body");
        }
    }
    // Override res.end to log response summary (duration updated when response finishes)
    const existingEnd = res.end;
    res.end = function (...args) {
        const duration = Date.now() - startTime;
        logger_1.logger.info({
            method: req.method,
            path: req.path,
            statusCode: res.statusCode,
            duration: `${duration}ms`,
        }, "[RESPONSE] Request completed");
        if (logger_1.logger.isLevelEnabled("debug")) {
            // attach additional details on debug as well
            logger_1.logger.debug({
                headers: res.getHeaders(),
            }, "[RESPONSE] Response headers");
        }
        return existingEnd.apply(res, args);
    };
    next();
};
exports.requestLoggerMiddleware = requestLoggerMiddleware;
/**
 * Body parsing error handler middleware
 * Catches JSON parsing errors with detailed logging
 */
const bodyParsingErrorHandler = (err, req, res, next) => {
    if (err instanceof SyntaxError && "body" in err) {
        logger_1.logger.error({
            method: req.method,
            path: req.path,
            error: err.message,
            statusCode: 400,
            hint: "Check if request body is valid JSON and Content-Type header is set to application/json",
        }, "[BODY-PARSER] JSON parsing failed");
        return res.status(400).json({
            success: false,
            message: "Invalid JSON in request body",
            error: err.message,
        });
    }
    next(err);
};
exports.bodyParsingErrorHandler = bodyParsingErrorHandler;
