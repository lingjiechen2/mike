import { Request, Response, NextFunction } from "express";

const SINGLE_USER_ID = "f3f6a836-a71a-41fd-a0e2-635ca5e5ccb3";
const SINGLE_USER_EMAIL = "local@mike.app";

export async function requireAuth(
  req: Request,
  res: Response,
  next: NextFunction,
): Promise<void> {
  res.locals.userId = SINGLE_USER_ID;
  res.locals.userEmail = SINGLE_USER_EMAIL;
  res.locals.token = "";
  next();
}
