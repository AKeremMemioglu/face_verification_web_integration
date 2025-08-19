// lib/api.ts

const BASE = "http://localhost:5000";

async function checkOk(res: Response) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} ${text}`.trim());
  }
  return res.json();
}

// ✅ Otosuggest
export async function suggestNames(q: string) {
  const res = await fetch(`${BASE}/api/suggest?q=${encodeURIComponent(q)}`);
  return checkOk(res);
}

// ✅ Model + isim için doğrulama
export async function verifyFace(model: string, name: string) {
  const res = await fetch(`${BASE}/api/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, name }),
  });
  return checkOk(res);
}

// ✅ Kişiye ait tüm resimleri getir
export async function getPersonImages(name: string) {
  const res = await fetch(`${BASE}/api/person_images?name=${encodeURIComponent(name)}`);
  return checkOk(res);
}

// ✅ Seçilen resim için doğrulama
export async function verifyByImage(model: string, image_name: string) {
  const res = await fetch(`${BASE}/api/verify_image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, image_name }),
  });
  return checkOk(res);
}
