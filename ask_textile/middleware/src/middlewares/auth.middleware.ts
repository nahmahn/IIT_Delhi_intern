import { Request, Response, NextFunction } from "express";
import { verifyToken } from "../utils/jwt";
import { AppError } from "../utils/AppError";
import { logger } from "../utils/logger";
import prisma from "../config/prisma";

export const authMiddleware = async (
  req: Request,
  res: Response,
  next: NextFunction,
) => {
  const authHeader = req.headers.authorization;

  logger.debug(
    { path: req.path, hasAuthHeader: !!authHeader },
    "[AUTH-MIDDLEWARE] Checking authentication",
  );

  if (!authHeader) {
    logger.warn(
      { path: req.path, method: req.method },
      "[AUTH-MIDDLEWARE] Missing authorization header",
    );
    throw new AppError("Unauthorized - No token provided", 401);
  }

  const parts = authHeader.split(" ");
  if (parts.length !== 2 || parts[0] !== "Bearer") {
    logger.warn(
      { path: req.path, authHeaderFormat: parts[0] },
      "[AUTH-MIDDLEWARE] Invalid authorization header format. Expected: Bearer <token>",
    );
    throw new AppError("Unauthorized - Invalid token format", 401);
  }

  const token = parts[1];
  if (!token) {
    logger.warn(
      { path: req.path },
      "[AUTH-MIDDLEWARE] Token missing after Bearer",
    );
    throw new AppError("Unauthorized - Invalid token format", 401);
  }

  try {
    const payload = verifyToken(token);
    const userId = payload.userId;

    // verify session exists and isn't expired
    const session = await prisma.session.findFirst({
      where: {
        userId,
        tokenHash: token,
      },
    });

    if (!session) {
      logger.warn(
        { userId, path: req.path },
        "[AUTH-MIDDLEWARE] Session not found for token",
      );
      throw new AppError("Unauthorized - Session not found", 401);
    }

    if (session.expiresAt < new Date()) {
      logger.warn(
        { userId, sessionId: session.id },
        "[AUTH-MIDDLEWARE] Session expired",
      );
      // delete expired session (best-effort)
      await prisma.session
        .deleteMany({ where: { id: session.id } })
        .catch(() => {});
      throw new AppError("Unauthorized - Session expired", 401);
    }

    // attach user and session to request for downstream handlers
    req.userId = userId;
    (req as any).sessionId = session.id;

    logger.debug(
      { userId, sessionId: session.id, path: req.path },
      "[AUTH-MIDDLEWARE] Token & session verified",
    );
    next();
  } catch (error) {
    logger.warn(
      { path: req.path, error: (error as Error).message },
      "[AUTH-MIDDLEWARE] Token verification failed",
    );
    throw new AppError("Unauthorized - Invalid token", 401);
  }
};
