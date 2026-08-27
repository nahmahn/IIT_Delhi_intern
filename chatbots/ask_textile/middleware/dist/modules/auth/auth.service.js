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
exports.AuthService = void 0;
const prisma_1 = __importDefault(require("../../config/prisma"));
const password_1 = require("../../utils/password");
const jwt_1 = require("../../utils/jwt");
const AppError_1 = require("../../utils/AppError");
const uuid_1 = require("uuid");
const logger_1 = require("../../utils/logger");
/**
 * Service for managing authentication.
 */
class AuthService {
    /**
     * Register a new user.
     * @param displayName - The user's display name.
     * @param email - The user's email.
     * @param password - The user's password.
     * @returns The created user object.
     */
    static register(displayName, email, password) {
        return __awaiter(this, void 0, void 0, function* () {
            logger_1.logger.debug({ email }, "Starting user registration");
            // Check if user already exists
            const existingUser = yield prisma_1.default.user.findUnique({
                where: { email },
            });
            if (existingUser) {
                logger_1.logger.warn({ email }, "Registration failed: User already exists");
                throw new AppError_1.AppError("User with this email already exists", 409);
            }
            const hashedPassword = yield (0, password_1.hashPassword)(password);
            const newUser = yield prisma_1.default.user.create({
                data: {
                    displayName,
                    email,
                    passwordHash: hashedPassword,
                },
                select: {
                    id: true,
                    email: true,
                    displayName: true,
                },
            });
            logger_1.logger.info({ userId: newUser.id, email }, "User registered successfully");
            return newUser;
        });
    }
    /**
     * Login to an existing user.
     * @param email - The user's email.
     * @param password - The user's password.
     * @returns A JWT token and user data.
     */
    static login(email, password) {
        return __awaiter(this, void 0, void 0, function* () {
            logger_1.logger.debug({ email }, "Login attempt started");
            const user = yield prisma_1.default.user.findUnique({
                where: { email },
            });
            if (!user) {
                logger_1.logger.warn({ email }, "Login failed: User not found");
                throw new AppError_1.AppError("Invalid credentials", 401);
            }
            const isValid = yield (0, password_1.comparePassword)(password, user.passwordHash);
            if (!isValid) {
                logger_1.logger.warn({ userId: user.id, email }, "Login failed: Invalid password");
                throw new AppError_1.AppError("Invalid credentials", 401);
            }
            // Update last login time
            yield prisma_1.default.user.update({
                where: { id: user.id },
                data: { lastLoginAt: new Date() },
            });
            logger_1.logger.debug({ userId: user.id }, "Last login updated");
            // Create a session
            const token = (0, jwt_1.signToken)(user.id);
            logger_1.logger.debug({ userId: user.id }, "JWT token generated");
            // Store session in database with auto-generated UUID
            const sessionId = (0, uuid_1.v4)();
            yield prisma_1.default.session.create({
                data: {
                    id: sessionId,
                    userId: user.id,
                    tokenHash: token, // In production, hash this
                    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
                },
            });
            logger_1.logger.debug({ userId: user.id, sessionId }, "Session created");
            const userResponse = {
                id: user.id,
                email: user.email,
                displayName: user.displayName,
            };
            logger_1.logger.info({ userId: user.id, email }, "User logged in successfully");
            return {
                token,
                user: userResponse,
            };
        });
    }
    /**
     * Logout a user by deleting their session.
     * @param userId - The user's ID.
     * @param token - The user's token.
     */
    static logout(userId, token) {
        return __awaiter(this, void 0, void 0, function* () {
            logger_1.logger.debug({ userId }, "Logout attempt started");
            const deletedSessions = yield prisma_1.default.session.deleteMany({
                where: {
                    userId,
                    tokenHash: token,
                },
            });
            logger_1.logger.info({ userId, deletedCount: deletedSessions.count }, "User logged out, sessions deleted");
        });
    }
    /**
     * Refresh user token.
     * @param token - The user's current token.
     * @returns New JWT token.
     */
    static refreshToken(token) {
        return __awaiter(this, void 0, void 0, function* () {
            logger_1.logger.debug("Token refresh attempt started");
            let payload;
            try {
                payload = (0, jwt_1.verifyToken)(token);
            }
            catch (error) {
                logger_1.logger.warn("Token refresh failed: Invalid token");
                throw new AppError_1.AppError("Invalid token", 401);
            }
            const userId = payload.userId;
            // Find user session
            const session = yield prisma_1.default.session.findFirst({
                where: {
                    userId,
                    tokenHash: token,
                },
            });
            if (!session) {
                logger_1.logger.warn({ userId }, "Token refresh failed: Session not found");
                throw new AppError_1.AppError("Session expired", 401);
            }
            if (session.expiresAt < new Date()) {
                logger_1.logger.warn({ userId, sessionId: session.id }, "Token refresh failed: Session expired");
                throw new AppError_1.AppError("Session expired", 401);
            }
            // Generate new token
            const newToken = (0, jwt_1.signToken)(userId);
            logger_1.logger.debug({ userId }, "New JWT token generated");
            // Update session with new token
            yield prisma_1.default.session.update({
                where: { id: session.id },
                data: {
                    tokenHash: newToken,
                    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
                },
            });
            logger_1.logger.info({ userId }, "Token refreshed successfully");
            return {
                token: newToken,
                refreshToken: newToken,
            };
        });
    }
    /**
     * Get current user by ID.
     * @param userId - The user's ID.
     * @returns User data.
     */
    static getCurrentUser(userId) {
        return __awaiter(this, void 0, void 0, function* () {
            logger_1.logger.debug({ userId }, "Fetching current user details");
            const user = yield prisma_1.default.user.findUnique({
                where: { id: userId },
                select: {
                    id: true,
                    email: true,
                    displayName: true,
                },
            });
            if (!user) {
                logger_1.logger.warn({ userId }, "User not found");
                throw new AppError_1.AppError("User not found", 404);
            }
            logger_1.logger.debug({ userId }, "Current user fetched successfully");
            return user;
        });
    }
}
exports.AuthService = AuthService;
