"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.env = void 0;
const zod_1 = require("zod");
const envSchema = zod_1.z.object({
    PORT: zod_1.z.string().default("3000"),
    DATABASE_URL: zod_1.z.string(),
    JWT_SECRET: zod_1.z.string(),
    BCRYPT_COST: zod_1.z.coerce.number().default(12),
    NODE_ENV: zod_1.z.enum(["development", "production"]).default("development"),
    // optional override; if omitted, logger decides based on NODE_ENV
    LOG_LEVEL: zod_1.z.string().optional(),
});
exports.env = envSchema.parse(process.env);
