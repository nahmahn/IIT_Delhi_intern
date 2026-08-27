import prisma from "../../config/prisma";
import { hashPassword, comparePassword } from "../../utils/password";
import { signToken, verifyToken } from "../../utils/jwt";
import { AppError } from "../../utils/AppError";
import { v4 as uuidv4 } from "uuid";
import { logger } from "../../utils/logger";
import {
  AuthUserResponse,
  LoginResponse,
  RefreshTokenResponse,
} from "./auth.types";

/**
 * Service for managing authentication.
 */
export class AuthService {
  /**
   * Register a new user.
   * @param displayName - The user's display name.
   * @param email - The user's email.
   * @param password - The user's password.
   * @returns The created user object.
   */
  static async register(displayName: string, email: string, password: string) {
    logger.debug({ email }, "Starting user registration");

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      logger.warn({ email }, "Registration failed: User already exists");
      throw new AppError("User with this email already exists", 409);
    }

    const hashedPassword = await hashPassword(password);
    const newUser = await prisma.user.create({
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

    logger.info({ userId: newUser.id, email }, "User registered successfully");
    return newUser;
  }

  /**
   * Login to an existing user.
   * @param email - The user's email.
   * @param password - The user's password.
   * @returns A JWT token and user data.
   */
  static async login(email: string, password: string): Promise<LoginResponse> {
    logger.debug({ email }, "Login attempt started");

    const user = await prisma.user.findUnique({
      where: { email },
    });

    if (!user) {
      logger.warn({ email }, "Login failed: User not found");
      throw new AppError("Invalid credentials", 401);
    }

    const isValid = await comparePassword(password, user.passwordHash);
    if (!isValid) {
      logger.warn({ userId: user.id, email }, "Login failed: Invalid password");
      throw new AppError("Invalid credentials", 401);
    }

    // Update last login time
    await prisma.user.update({
      where: { id: user.id },
      data: { lastLoginAt: new Date() },
    });
    logger.debug({ userId: user.id }, "Last login updated");

    // Create a session
    const token = signToken(user.id);
    logger.debug({ userId: user.id }, "JWT token generated");
    
    // Store session in database with auto-generated UUID
    const sessionId = uuidv4();
    await prisma.session.create({
      data: {
        id: sessionId,
        userId: user.id,
        tokenHash: token, // In production, hash this
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
      },
    });
    logger.debug({ userId: user.id, sessionId }, "Session created");

    const userResponse: AuthUserResponse = {
      id: user.id,
      email: user.email,
      displayName: user.displayName,
    };

    logger.info({ userId: user.id, email }, "User logged in successfully");
    return {
      token,
      user: userResponse,
    };
  }

  /**
   * Logout a user by deleting their session.
   * @param userId - The user's ID.
   * @param token - The user's token.
   */
  static async logout(userId: string, token: string): Promise<void> {
    logger.debug({ userId }, "Logout attempt started");

    const deletedSessions = await prisma.session.deleteMany({
      where: {
        userId,
        tokenHash: token,
      },
    });

    logger.info(
      { userId, deletedCount: deletedSessions.count },
      "User logged out, sessions deleted",
    );
  }

  /**
   * Refresh user token.
   * @param token - The user's current token.
   * @returns New JWT token.
   */
  static async refreshToken(token: string): Promise<RefreshTokenResponse> {
    logger.debug("Token refresh attempt started");

    let payload;
    try {
      payload = verifyToken(token);
    } catch (error) {
      logger.warn("Token refresh failed: Invalid token");
      throw new AppError("Invalid token", 401);
    }

    const userId = payload.userId;

    // Find user session
    const session = await prisma.session.findFirst({
      where: {
        userId,
        tokenHash: token,
      },
    });

    if (!session) {
      logger.warn({ userId }, "Token refresh failed: Session not found");
      throw new AppError("Session expired", 401);
    }

    if (session.expiresAt < new Date()) {
      logger.warn(
        { userId, sessionId: session.id },
        "Token refresh failed: Session expired",
      );
      throw new AppError("Session expired", 401);
    }

    // Generate new token
    const newToken = signToken(userId);
    logger.debug({ userId }, "New JWT token generated");

    // Update session with new token
    await prisma.session.update({
      where: { id: session.id },
      data: {
        tokenHash: newToken,
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000),
      },
    });

    logger.info({ userId }, "Token refreshed successfully");
    return {
      token: newToken,
      refreshToken: newToken,
    };
  }

  /**
   * Get current user by ID.
   * @param userId - The user's ID.
   * @returns User data.
   */
  static async getCurrentUser(userId: string): Promise<AuthUserResponse> {
    logger.debug({ userId }, "Fetching current user details");

    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        displayName: true,
      },
    });

    if (!user) {
      logger.warn({ userId }, "User not found");
      throw new AppError("User not found", 404);
    }

    logger.debug({ userId }, "Current user fetched successfully");
    return user;
  }
}
