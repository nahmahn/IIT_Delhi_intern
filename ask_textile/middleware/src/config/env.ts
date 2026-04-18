import { z } from "zod";
const envSchema = z.object({
  PORT: z.string().default("3000"),
  DATABASE_URL: z.string(),
  JWT_SECRET: z.string(),
  BCRYPT_COST: z.coerce.number().default(12),
  NODE_ENV: z.enum(["development", "production"]).default("development"),
  // optional override; if omitted, logger decides based on NODE_ENV
  LOG_LEVEL: z.string().optional(),
});

export const env = envSchema.parse(process.env);
