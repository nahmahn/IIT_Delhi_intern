"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = require("express");
const auth_controller_1 = require("./auth.controller");
const asyncHandler_1 = require("../../utils/asyncHandler");
const auth_middleware_1 = require("../../middlewares/auth.middleware");
const logger_1 = require("../../utils/logger");
// Request logging middleware
const requestLogger = (req, res, next) => {
    logger_1.logger.info({ method: req.method, path: req.path, ip: req.ip }, "[AUTH-ROUTE] Incoming request");
    next();
};
const router = (0, express_1.Router)();
router.use(requestLogger);
// Public routes
router.post("/register", (0, asyncHandler_1.asyncHandler)(auth_controller_1.AuthController.register));
router.post("/login", (0, asyncHandler_1.asyncHandler)(auth_controller_1.AuthController.login));
router.post("/refresh", (0, asyncHandler_1.asyncHandler)(auth_controller_1.AuthController.refreshToken));
// Protected routes
router.get("/me", auth_middleware_1.authMiddleware, (0, asyncHandler_1.asyncHandler)(auth_controller_1.AuthController.getCurrentUser));
router.post("/logout", auth_middleware_1.authMiddleware, (0, asyncHandler_1.asyncHandler)(auth_controller_1.AuthController.logout));
exports.default = router;
