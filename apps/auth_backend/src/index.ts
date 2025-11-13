import express from "express";
import jwt from "jsonwebtoken";
import bcrypt from "bcrypt";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { dotenv_path } from "@repo/dotenv-path";
import { signupSchema, signinSchema } from "@repo/zod-scemma";
import { prismaClient } from "@repo/db";
// Get the directory name from the current module URL
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from .env file
dotenv.config({ path: dotenv_path });

const app = express();
app.use(express.json());
app.use(
  cors({
    origin: "http://localhost:3000",
  })
);

app.get("/health", (req, res) => {
  res.json({ status: "ok" });
});

app.post("/signup", async (req, res) => {
  const { name, email, password, degree, field, university, graduationYear } =
    req.body;
  if (
    !name ||
    !email ||
    !password ||
    !degree ||
    !field ||
    !university ||
    !graduationYear
  ) {
    return res.status(400).json({ message: "Missing required fields" });
  }

  try {
    // Validate input using zod schema
    const result = await signupSchema.safeParseAsync({
      name: name,
      email,
      password,
      degree,
      field,
      university,
      graduationYear,
    });
    if (!result.success) {
      return res
        .status(400)
        .json({ message: "Invalid input", errors: result.error });
    }
  } catch (error) {
    console.error("Error validating signup data:", error);
    return res.status(500).json({ message: "Internal server error" });
  }

  //encrypting the password
  if (process.env.SALT_ROUNDS === undefined) {
    return res.status(500).json({ message: "Internal server error" });
  }

  const existingUser = await prismaClient.user.findUnique({
    where: {
      email: email,
    },
  });
  if (existingUser) {
    return res.status(409).json({ message: "User already exists" });
  }
  const hashedPassword = await bcrypt.hash(
    password,
    Number(process.env.SALT_ROUNDS)
  );
  // add the user

  try {
    const user = await prismaClient.user.create({
      data: {
        email: email,
        name: name,
        password: hashedPassword,
        degree: degree,
        field: field,
        university: university,
        graduationYear: graduationYear,
      },
    });

    if (!user) {
      res.status(500).json({
        message: "internal server error",
      });
    }
    res.status(201).json({
      message: "user created successfully",
    });
  } catch (error) {
    res.status(500).json({
      error: error,
    });
  }
});

app.post("/signin", async (req, res) => {
  const { email, password } = req.body;
  if (!email || !password) {
    return res.status(400).json({ message: "Missing required fields" });
  }

  try {
    const result = await signinSchema.safeParseAsync({
      email,
      password,
    });
    if (!result.success) {
      return res
        .status(400)
        .json({ message: "Invalid input", errors: result.error });
    }
  } catch (error) {
    console.error("Error validating signin data:", error);
    return res.status(500).json({ message: "Internal server error" });
  }
  try {
    const user = await prismaClient.user.findUnique({
      where: {
        email: email,
      },
    });
    if (!user) {
      return res.status(401).json({ message: "Invalid email or password" });
    }
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(401).json({ message: "Invalid email or password" });
    }
    if (process.env.JWT_SECRET === undefined) {
      return res.status(500).json({ message: "Internal server error" });
    }
    const token = jwt.sign(
      {
        sub: user.id, // Standard "subject" claim — user’s unique ID
        email: user.email, // Optional, for convenience
        scope: ["train", "predict"], // Optional array of permissions
      },
      process.env.JWT_SECRET, // Same as AUTH_JWT_SECRET in FastAPI
      {
        algorithm: "HS256",
        issuer: "http://localhost:4000", // Must match AUTH_JWT_ISSUER
        audience: "mlapp", // Must match AUTH_JWT_AUDIENCE
        expiresIn: "1h", // Expiration (required)
      }
    );
    res.status(200).json({ token: token, userId: user.id, name: user.name });
  } catch (error) {
    console.error("Error during signin:", error);
    res.status(500).json({ message: "Internal server error" });
  }
});

app.listen(process.env.AUTH_PORT, () => {
  console.log(`Auth server listening on port ${process.env.AUTH_PORT}`);
});
