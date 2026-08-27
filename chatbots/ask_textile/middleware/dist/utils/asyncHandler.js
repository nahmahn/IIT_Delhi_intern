"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.asyncHandler = void 0;
/**
 * A middleware function that wraps an asynchronous function and handles any errors that occur.
 *
 * @param {Function} fn - The asynchronous function to wrap.
 * @returns {Function} - A middleware function that calls the wrapped function and handles any errors that occur.
 */
const asyncHandler = (fn) => (req, res, next) => {
    /**
     * Call the wrapped function and handle any errors that occur.
     * @param {Request} req - The request object.
     * @param {Response} res - The response object.
     * @param {NextFunction} next - The next function to call if an error occurs.
     * @returns {Promise<void>} - A promise that resolves when the wrapped function is called successfully or rejects when an error occurs.
     */
    return Promise.resolve(fn(req, res, next)).catch(next);
};
exports.asyncHandler = asyncHandler;
