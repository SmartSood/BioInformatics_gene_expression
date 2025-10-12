import express  from "express";
import jwt from "jsonwebtoken";
export const authMiddleware = (req: express.Request, res: express.Response, next: express.NextFunction) => {
    const authHeader = req.headers.token;
    if (!authHeader) {
        return res.status(401).json({ message: "Token header missing" });
    }
    const token = req.headers.token as string;
    if (!token) {
        return res.status(401).json({ message: "Token missing" });
    }
    
    if (process.env.JWT_SECRET === undefined) {
        return res.status(500).json({ message: "Internal server error" });
    }
    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET) as { userId: number };
        (req as any).userId = decoded.userId;
    }
    catch (error) {
        return res.status(401).json({ message: "Invalid token" });

  next();
}
};
