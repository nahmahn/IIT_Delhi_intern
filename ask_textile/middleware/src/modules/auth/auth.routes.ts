import { Router, Request, Response, NextFunction } from "express";
import { AuthController } from "./auth.controller";
import { asyncHandler } from "../../utils/asyncHandler";
import { authMiddleware } from "../../middlewares/auth.middleware";
import { logger } from "../../utils/logger";

// Request logging middleware
const requestLogger = (req: Request, res: Response, next: NextFunction) => {
  logger.info(
    { method: req.method, path: req.path, ip: req.ip },
    "[AUTH-ROUTE] Incoming request",
  );
  next();
};

const router = Router();
router.use(requestLogger);

// Public routes
router.post("/register", asyncHandler(AuthController.register));
router.post("/login", asyncHandler(AuthController.login));
router.post("/refresh", asyncHandler(AuthController.refreshToken));

// Protected routes
router.get("/me", authMiddleware, asyncHandler(AuthController.getCurrentUser));
router.post("/logout", authMiddleware, asyncHandler(AuthController.logout));

export default router;
