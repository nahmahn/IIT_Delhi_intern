import express from "express";
import authRoutes from "./modules/auth/auth.routes";
import chatRoutes from "./modules/chat/chat.routes";
import { errorMiddleware } from "./middlewares/error.middleware";
import { corsConfig } from "./config/cors";
import {
  requestLoggerMiddleware,
  bodyParsingErrorHandler,
} from "./middlewares/request-logger.middleware";

const app = express();
app.use(corsConfig);
app.use(express.json());

// Body parsing error handler (catches JSON parsing errors)
app.use(bodyParsingErrorHandler);

// Request/Response logging middleware
app.use(requestLoggerMiddleware);

app.use("/api/v1/auth", authRoutes);
app.use("/api/v1/chat", chatRoutes);

//! Global error handler (ALWAYS LAST)
app.use(errorMiddleware);

export default app;
