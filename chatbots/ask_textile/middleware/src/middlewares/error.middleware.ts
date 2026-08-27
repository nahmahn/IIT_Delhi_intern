import { Request, Response, NextFunction } from "express";
import { AppError } from "../utils/AppError";
import { logger } from "../utils/logger";
import { ZodError } from "zod";

export const errorMiddleware = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction,
) => {
  if (err instanceof ZodError) {
    const errors = err.issues.map((e) => ({
      path: e.path.join("."),
      message: e.message,
    }));
    logger.warn(
      { endpoint: req.path, method: req.method, errors },
      "[ERROR] Validation error",
    );
    return res.status(400).json({
      success: false,
      message: "Validation Error",
      errors,
    });
  }

  if (err instanceof AppError) {
    logger.warn(
      {
        endpoint: req.path,
        method: req.method,
        statusCode: err.statusCode,
        message: err.message,
      },
      "[ERROR] Application error",
    );
    return res.status(err.statusCode).json({
      success: false,
      message: err.message,
    });
  }

  // always log at least the message; include stack only when debug
  const baseError: any = {
    endpoint: req.path,
    method: req.method,
    errorMessage: err.message,
  };
  if (logger.isLevelEnabled("debug")) {
    baseError.stack = err.stack;
  }

  logger.error(baseError, "[ERROR] Unhandled error");

  return res.status(500).json({
    success: false,
    message: "Internal Server Error",
  });
};
