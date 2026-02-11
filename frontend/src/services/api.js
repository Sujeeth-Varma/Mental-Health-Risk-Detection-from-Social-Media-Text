import axios from "axios";

const api = axios.create({
  baseURL: "",
  headers: { "Content-Type": "application/json" },
  timeout: 30000,
});

export async function predictRisk(text) {
  const { data } = await api.post("/predict", { text });
  return data;
}

export async function analyzeText(text) {
  const { data } = await api.post("/analyze", { text });
  return data;
}

export async function checkHealth() {
  const { data } = await api.get("/health");
  return data;
}

export default api;
