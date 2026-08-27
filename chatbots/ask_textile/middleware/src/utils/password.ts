import bcrypt from "bcrypt";
import {env} from "../config/env"

// ? commen function for hashing the password
export const hashPassword = async (password:string):Promise<string>=>{
   return  bcrypt.hash(password,env.BCRYPT_COST);
}

// ? commen function for comparingPassword
export const comparePassword = async (password:string,hash:string):Promise<boolean>=>{
return bcrypt.compare(password,hash);
}
