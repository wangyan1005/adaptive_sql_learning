import axios from 'axios'

export const api = axios.create({
    baseURL: "https://adaptive-sql-learning.onrender.com",   
    // baseURL: "http://127.0.0.1:3001",
});
