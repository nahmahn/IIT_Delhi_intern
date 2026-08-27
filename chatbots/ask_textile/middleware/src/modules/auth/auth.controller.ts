import { Request, Response } from "express";
import { AuthService } from "./auth.service";
import { logger } from "../../utils/logger";
import { AppError } from "../../utils/AppError";
import {
  RegisterRequest,
  LoginRequest,
  AuthUserResponse,
  LoginResponse,
  RefreshTokenRequest,
  LogoutResponse,
} from "./auth.types";
import { registerSchema, loginSchema, refreshTokenSchema } from "./auth.schema";

/**
 * Controller for managing authentication.
 */
export class AuthController {
  /**
   * Register a new user.
   *
   * @param req - The request object.
   * @param res - The response object.
   *
   * @remarks
   * This endpoint is used to register a new user. It takes the user's displayName, email and password as parameters and returns the created user object.
   */
  static async register(req: Request, res: Response) {
    const body: RegisterRequest = registerSchema.parse(req.body);
    const { displayName, email, password } = body;

    logger.info({ email }, "[AUTH] Register endpoint called");

    const user: AuthUserResponse = await AuthService.register(
      displayName,
      email,
      password,
    );

    logger.info(
      { userId: user.id, email },
      "[AUTH] User registration response sent",
    );
    res.status(201).json(user);
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
  static async login(req: Request, res: Response) {
    const body: LoginRequest = loginSchema.parse(req.body);
    const { email, password } = body;

    logger.info({ email }, "[AUTH] Login endpoint called");

    const response: LoginResponse = await AuthService.login(email, password);

    logger.info(
      { userId: response.user.id, email },
      "[AUTH] Login response sent",
    );
    res.json(response);
  }

  /**
   * Logout the current user.
   *
   * @param req - The request object.
   * @param res - The response object.
   */
  static async logout(req: Request, res: Response) {
    const userId = req.userId;
    if (!userId) {
      throw new AppError("Unauthorized", 401);
    }
    const authHeader = req.headers.authorization;
    const token = authHeader?.split(" ")[1] || "";

    logger.info({ userId }, "[AUTH] Logout endpoint called");

    await AuthService.logout(userId, token);
    const response: LogoutResponse = { message: "Logout successful" };

    logger.info({ userId }, "[AUTH] Logout response sent");
    res.json(response);
  }

  /**
   * Refresh user token.
   *
   * @param req - The request object.
   * @param res - The response object.
   */
  static async refreshToken(req: Request, res: Response) {
    const body: RefreshTokenRequest = refreshTokenSchema.parse(req.body);

    logger.info("[AUTH] Refresh token endpoint called");

    const response = await AuthService.refreshToken(body.refreshToken);

    logger.info("[AUTH] Token refresh response sent");
    res.json(response);
  }

  /**
   * Get current authenticated user.
   *
   * @param req - The request object.
   * @param res - The response object.
   */
  static async getCurrentUser(req: Request, res: Response) {
    const userId = req.userId;
    if (!userId) {
      throw new AppError("Unauthorized", 401);
    }

    logger.info({ userId }, "[AUTH] Get current user endpoint called");

    const user: AuthUserResponse = await AuthService.getCurrentUser(userId);

    logger.info({ userId }, "[AUTH] Current user response sent");
    res.json(user);
  }
}
