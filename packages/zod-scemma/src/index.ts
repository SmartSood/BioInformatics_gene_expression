import {z} from 'zod';



const signupSchema = z.object({
  email: z.string().email(),
  name: z.string().min(2).max(100),
  password: z.string().min(6).max(100),
  degree: z.enum(['Bachelors', 'Masters', 'PhD', 'Other']),
  field: z.string().min(2).max(100),
  university: z.string().min(2).max(100),
  graduationYear: z.number().min(1900).max(new Date().getFullYear() + 10),
});
  
const signinSchema = z.object({
  email: z.string().email(),
  password: z.string().min(6).max(100),
});
export {signupSchema, signinSchema};