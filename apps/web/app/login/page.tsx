"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  Microscope,
  Mail,
  Lock,
  User,
  GraduationCap,
  BookOpen,
  Calendar,
} from "lucide-react";
import { degree_type } from "@repo/types";
import axios from "axios";
import { AUTH_BACKEND_URL } from "@repo/config";

export default function LoginPage() {
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [degree, setDegree] = useState<degree_type>("Bachelors");
  const [field, setField] = useState("");
  const [university, setUniversity] = useState("");
  const [graduationYear, setGraduationYear] = useState<number | "">("");
  const [loading, setLoading] = useState(false);
  const [confirmPassword, setConfirmPassword] = useState("");
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const baseUrl = AUTH_BACKEND_URL;
      if (baseUrl === undefined) {
        alert("Backend URL server error.");
        setLoading(false);
        return;
      }
      if (isSignUp) {
        if (password !== confirmPassword) {
          alert("Passwords do not match");
          setLoading(false);
          return;
        }

        // Call backend signup API
        const response = await axios.post(
          `${baseUrl}/signup`,
          {
            email,
            password,
            name,
            degree,
            field,
            university,
            graduationYear,
          },
          { headers: { "Content-Type": "application/json" } }
        );

        if (response.status === 201) {
          alert("Signup successful!");
          router.push("/dashboard");
        } else if (response.status === 409) {
          alert("User already exists Please sign in");
          setLoading(false);
          return;
        } else {
          alert("Signup failed. Please try again.");
          console.log("Signup failed", response);
          setLoading(false);
          return;
        }
      } else {
        // Call backend signin API
        const response = await axios.post<{
          name: any;
          userId: any; token: string 
}>(
          `${baseUrl}/signin`,
          { email, password },
          { headers: { "Content-Type": "application/json" } }
        );

        const user = {
          email: email,
          id: response.data.userId,
          name: response.data.name,
        };

        // Store user details in sessionStorage for dashboard access
        sessionStorage.setItem("authToken", response.data.token);
        sessionStorage.setItem("userId", response.data.userId);
        sessionStorage.setItem("User", JSON.stringify(user));
        alert("Signin successful!");
        router.push("/dashboard");
      }
    } catch (error: any) {
      if (
        error.response &&
        error.response.data &&
        error.response.data.message
      ) {
        alert(error.response?.data?.message || "Authentication failed.");
      } else {
        alert("Authentication failed. Please try again.");
      }
      console.error("Auth error:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl justify-center w-full font-bold text-white mb-2">
            {isSignUp ? "Create Account" : "Welcome Back!"}
          </h1>
        </div>

        <div className="bg-gradient-to-br from-slate-800 to-slate-900 rounded-xl shadow-2xl border border-slate-700/50 p-8">
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => setIsSignUp(false)}
              className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                !isSignUp
                  ? "bg-gradient-to-r from-teal-600 to-blue-600 text-white"
                  : "bg-slate-700/30 text-slate-400 hover:bg-slate-700/50"
              }`}
            >
              Sign In
            </button>
            <button
              onClick={() => setIsSignUp(true)}
              className={`flex-1 py-2 rounded-lg font-medium transition-all ${
                isSignUp
                  ? "bg-gradient-to-r from-teal-600 to-blue-600 text-white"
                  : "bg-slate-700/30 text-slate-400 hover:bg-slate-700/50"
              }`}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {isSignUp && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                    className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                    placeholder="Enter your name"
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                  placeholder="Enter your email"
                />
              </div>
            </div>

            {isSignUp && (
              <div>
                <div className="flex gap-4">
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Degree
                    </label>
                    <div className="relative">
                      <Microscope className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <select
                        value={degree}
                        onChange={(e) =>
                          setDegree(e.target.value as degree_type)
                        }
                        required
                        className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                      >
                        <option value="Bachelors">Bachelors</option>
                        <option value="Masters">Masters</option>
                        <option value="PhD">PhD</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                  </div>

                  <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      University
                    </label>
                    <div className="relative">
                      <GraduationCap className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <input
                        type="text"
                        value={university}
                        onChange={(e) => setUniversity(e.target.value)}
                        required
                        className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                        placeholder="IIT BHU"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex gap-4">
                  <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Field of Study
                    </label>
                    <div className="relative">
                      <BookOpen className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <input
                        type="text"
                        value={field}
                        onChange={(e) => setField(e.target.value)}
                        required
                        className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                        placeholder="BioMedical Engineering"
                      />
                    </div>
                  </div>

                  <div className="flex-1">
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Graduation Year
                    </label>
                    <div className="relative">
                      <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                      <input
                        type="number"
                        value={graduationYear}
                        onChange={(e) =>
                          setGraduationYear(
                            e.target.value ? parseInt(e.target.value) : ""
                          )
                        }
                        required
                        className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                        placeholder="e.g., 2025"
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                  placeholder="Enter your password"
                />
              </div>
            </div>

            {isSignUp && (
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Confirm Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    className="w-full pl-11 pr-4 py-3 bg-slate-700/50 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-teal-500/50 focus:border-teal-500/50"
                    placeholder="Confirm your password"
                  />
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-500 hover:to-blue-500 text-white rounded-lg font-medium transition-all shadow-lg shadow-teal-500/20 hover:shadow-teal-500/40 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Processing..." : isSignUp ? "Sign Up" : "Sign In"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
