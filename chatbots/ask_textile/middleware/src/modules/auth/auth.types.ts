export interface RegisterRequest {
  displayName: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface AuthUserResponse {
  id: string;
  email: string;
  displayName: string | null;
}

export interface LoginResponse {
  token: string;
  refreshToken?: string;
  user: AuthUserResponse;
}

export interface RefreshTokenRequest {
  refreshToken: string;
}

export interface RefreshTokenResponse {
  token: string;
  refreshToken: string;
}

export interface LogoutResponse {
  message: string;
}
