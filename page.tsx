"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { verifyFace, getPersonImages, verifyByImage, suggestNames } from "@/lib/api";

type ImgRes = { name: string; url: string };
type TopItem = { name: string; similarity: number; embedding_time: number; url: string };
type VerifyResp = {
  match: boolean;
  model: string;
  target: ImgRes;
  top3: TopItem[];
  message: string;
};

export default function Home() {
  const [model, setModel] = useState("");
  const [manualName, setManualName] = useState("");
  const [result, setResult] = useState("");
  const [images, setImages] = useState<{ target: ImgRes; top3: TopItem[] } | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [personImages, setPersonImages] = useState<ImgRes[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    document.title = "Face Recognition and Verification Project";
  }, []);

  const fetchSuggestions = async (value: string) => {
    if (!value) {
      setSuggestions([]);
      return;
    }
    try {
      const data = await suggestNames(value);
      setSuggestions(data);
    } catch {
      setSuggestions([]);
    }
  };

  const fetchPersonImagesHandler = async (name: string) => {
    try {
      const res = await getPersonImages(name);
      setPersonImages(res.images);
      setSelectedImage(null);
      setImages(null);
    } catch {
      setPersonImages([]);
    }
  };

  const handleVerifyImage = async (imgName: string) => {
    if (!model) {
      setResult("⚠️ Önce bir model seçin.");
      return;
    }
    setLoading(true);
    try {
      const response: VerifyResp = await verifyByImage(model, imgName);
      setResult(`✅ ${response.message}`);
      setImages({ target: response.target, top3: response.top3 });
      setSelectedImage(imgName);
    } catch (err: any) {
      console.error("verifyByImage error:", err);
      setResult("❌ Resim doğrulama başarısız.");
    } finally {
      setLoading(false);
    }
  };

  const handleVerify = async () => {
    const name = manualName.trim();
    if (!model || !name) {
      setResult("⚠️ Model ve isim girin.");
      return;
    }
    setLoading(true);
    try {
      const response: VerifyResp = await verifyFace(model, name);
      setResult(`✅ ${response.message}`);
      setImages({ target: response.target, top3: response.top3 });
    } catch (err: any) {
      console.error("verifyFace error:", err);
      setResult("❌ API bağlantısı başarısız oldu.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-6 bg-blue-100">
      <h1 className="text-3xl font-bold mb-8">Face Recognition and Verification Project</h1>

      <Card className="w-full max-w-6xl shadow-lg p-6 border-none bg-white">
        <CardContent className="flex flex-col items-center text-center space-y-6">

          {/* Form + Auto-suggest */}
          <div className="flex flex-col items-center space-y-2 relative">
            <label className="font-semibold text-sm">Model Seçin:</label>
            <select
              className="border p-1 rounded text-sm h-8 max-w-[220px]"
              onChange={(e) => setModel(e.target.value)}
              value={model}
            >
              <option value="">Model Seçin</option>
              <option value="VGG-Face">VGG-Face</option>
              <option value="Facenet">Facenet</option>
              <option value="ArcFace">ArcFace</option>
              <option value="KeremNet">KeremNet</option>{/* <-- eklendi */}
            </select>

            <label className="font-semibold text-sm mt-1">Kişi Adı:</label>
            <div className="relative w-full max-w-[220px]">
              <Input
                placeholder="Kişi adını yazın"
                value={manualName}
                onChange={async (e) => {
                  const v = e.target.value;
                  setManualName(v);
                  await fetchSuggestions(v);
                }}
                className="text-sm h-8 w-full"
              />
              {suggestions.length > 0 && (
                <ul className="absolute z-10 left-0 right-0 mt-1 max-h-64 overflow-auto rounded border bg-white text-left text-sm shadow">
                  {suggestions.map((s) => (
                    <li
                      key={s}
                      onClick={() => {
                        setManualName(s);
                        fetchPersonImagesHandler(s);
                        setSuggestions([]);
                      }}
                      className="cursor-pointer px-2 py-1 hover:bg-gray-100"
                    >
                      {s}
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <Button
              onClick={handleVerify}
              disabled={loading}
              className="mt-3 text-xs px-3 py-1 max-w-[120px] hover:scale-105 transition-transform"
            >
              {loading ? "Çalışıyor..." : "Eşleştir"}
            </Button>

            {result && <p className="text-green-700 font-semibold text-sm mt-2">{result}</p>}
          </div>

          {/* Kişiye ait resimler (liste) */}
          {personImages.length > 0 && !selectedImage && (
            <div className="mt-4 text-center">
              <h2 className="text-lg font-semibold mb-2">Bir resim seç:</h2>
              <ul className="flex flex-col items-center space-y-2">
                {personImages.map((img) => (
                  <li
                    key={img.name}
                    onClick={() => handleVerifyImage(img.name)}
                    className="cursor-pointer px-4 py-1 border rounded bg-gray-100 hover:bg-blue-200 w-64 text-sm"
                  >
                    {img.name}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Sonuçlar */}
          {images && (
            <div className="flex flex-wrap justify-center gap-10 mt-6">
              <div className="w-52 text-center">
                <img src={images.target.url} className="w-48 h-48 object-cover rounded" />
                <p className="font-bold mt-2 text-sm">{images.target.name}</p>
              </div>
              {images.top3.map((img) => (
                <div key={img.name} className="w-52 text-center">
                  <img src={img.url} className="w-48 h-48 object-cover rounded" />
                  <p className="mt-2 text-sm">{img.name}</p>
                  <p className="text-green-600 font-semibold text-xs">{img.similarity}%</p>
                  <p className="text-blue-600 font-medium text-xs">⏱ {img.embedding_time}s</p>
                </div>
              ))}
            </div>
          )}

        </CardContent>
      </Card>
    </div>
  );
}
