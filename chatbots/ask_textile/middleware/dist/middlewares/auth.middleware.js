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
exports.authMiddleware = void 0;
const jwt_1 = require("../utils/jwt");
const AppError_1 = require("../utils/AppError");
const logger_1 = require("../utils/logger");
const prisma_1 = __importDefault(require("../config/prisma"));
const authMiddleware = (req, res, next) => __awaiter(void 0, void 0, void 0, function* () {
    const authHeader = req.headers.authorization;
    logger_1.logger.debug({ path: req.path, hasAuthHeader: !!authHeader }, "[AUTH-MIDDLEWARE] Checking authentication");
    if (!authHeader) {
        logger_1.logger.warn({ path: req.path, method: req.method }, "[AUTH-MIDDLEWARE] Missing authorization header");
        throw new AppError_1.AppError("Unauthorized - No token provided", 401);
    }
    const parts = authHeader.split(" ");
    if (parts.length !== 2 || parts[0] !== "Bearer") {
        logger_1.logger.warn({ path: req.path, authHeaderFormat: parts[0] }, "[AUTH-MIDDLEWARE] Invalid authorization header format. Expected: Bearer <token>");
        throw new AppError_1.AppError("Unauthorized - Invalid token format", 401);
    }
    const token = parts[1];
    if (!token) {
        logger_1.logger.warn({ path: req.path }, "[AUTH-MIDDLEWARE] Token missing after Bearer");
        throw new AppError_1.AppError("Unauthorized - Invalid token format", 401);
    }
    try {
        const payload = (0, jwt_1.verifyToken)(token);
        const userId = payload.userId;
        // verify session exists and isn't expired
        const session = yield prisma_1.default.session.findFirst({
            where: {
                userId,
                tokenHash: token,
            },
        });
        if (!session) {
            logger_1.logger.warn({ userId, path: req.path }, "[AUTH-MIDDLEWARE] Session not found for token");
            throw new AppError_1.AppError("Unauthorized - Session not found", 401);
        }
        if (session.expiresAt < new Date()) {
            logger_1.logger.warn({ userId, sessionId: session.id }, "[AUTH-MIDDLEWARE] Session expired");
            // delete expired session (best-effort)
            yield prisma_1.default.session
                .deleteMany({ where: { id: session.id } })
                .catch(() => { });
            throw new AppError_1.AppError("Unauthorized - Session expired", 401);
        }
        // attach user and session to request for downstream handlers
        req.userId = userId;
        req.sessionId = session.id;
        logger_1.logger.debug({ userId, sessionId: session.id, path: req.path }, "[AUTH-MIDDLEWARE] Token & session verified");
        next();
    }
    catch (error) {
        logger_1.logger.warn({ path: req.path, error: error.message }, "[AUTH-MIDDLEWARE] Token verification failed");
        throw new AppError_1.AppError("Unauthorized - Invalid token", 401);
    }
});
exports.authMiddleware = authMiddleware;
