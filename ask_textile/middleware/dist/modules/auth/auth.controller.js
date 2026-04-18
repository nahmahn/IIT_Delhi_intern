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
Object.defineProperty(exports, "__esModule", { value: true });
exports.AuthController = void 0;
const auth_service_1 = require("./auth.service");
const logger_1 = require("../../utils/logger");
const AppError_1 = require("../../utils/AppError");
const auth_schema_1 = require("./auth.schema");
/**
 * Controller for managing authentication.
 */
class AuthController {
    /**
     * Register a new user.
     *
     * @param req - The request object.
     * @param res - The response object.
     *
     * @remarks
     * This endpoint is used to register a new user. It takes the user's displayName, email and password as parameters and returns the created user object.
     */
    static register(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const body = auth_schema_1.registerSchema.parse(req.body);
            const { displayName, email, password } = body;
            logger_1.logger.info({ email }, "[AUTH] Register endpoint called");
            const user = yield auth_service_1.AuthService.register(displayName, email, password);
            logger_1.logger.info({ userId: user.id, email }, "[AUTH] User registration response sent");
            res.status(201).json(user);
        });
    }
    /**
     * Login to an existing user.
     *
     * @param req - The request object.
     * @param res - The response object.
     *
     * @remarks
     * This endpoint is used to login to an existing user. It takes the user's email and password as parameters and returns a JWT token.
     */
    static login(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const body = auth_schema_1.loginSchema.parse(req.body);
            const { email, password } = body;
            logger_1.logger.info({ email }, "[AUTH] Login endpoint called");
            const response = yield auth_service_1.AuthService.login(email, password);
            logger_1.logger.info({ userId: response.user.id, email }, "[AUTH] Login response sent");
            res.json(response);
        });
    }
    /**
     * Logout the current user.
     *
     * @param req - The request object.
     * @param res - The response object.
     */
    static logout(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const userId = req.userId;
            if (!userId) {
                throw new AppError_1.AppError("Unauthorized", 401);
            }
            const authHeader = req.headers.authorization;
            const token = (authHeader === null || authHeader === void 0 ? void 0 : authHeader.split(" ")[1]) || "";
            logger_1.logger.info({ userId }, "[AUTH] Logout endpoint called");
            yield auth_service_1.AuthService.logout(userId, token);
            const response = { message: "Logout successful" };
            logger_1.logger.info({ userId }, "[AUTH] Logout response sent");
            res.json(response);
        });
    }
    /**
     * Refresh user token.
     *
     * @param req - The request object.
     * @param res - The response object.
     */
    static refreshToken(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const body = auth_schema_1.refreshTokenSchema.parse(req.body);
            logger_1.logger.info("[AUTH] Refresh token endpoint called");
            const response = yield auth_service_1.AuthService.refreshToken(body.refreshToken);
            logger_1.logger.info("[AUTH] Token refresh response sent");
            res.json(response);
        });
    }
    /**
     * Get current authenticated user.
     *
     * @param req - The request object.
     * @param res - The response object.
     */
    static getCurrentUser(req, res) {
        return __awaiter(this, void 0, void 0, function* () {
            const userId = req.userId;
            if (!userId) {
                throw new AppError_1.AppError("Unauthorized", 401);
            }
            logger_1.logger.info({ userId }, "[AUTH] Get current user endpoint called");
            const user = yield auth_service_1.AuthService.getCurrentUser(userId);
            logger_1.logger.info({ userId }, "[AUTH] Current user response sent");
            res.json(user);
        });
    }
}
exports.AuthController = AuthController;
