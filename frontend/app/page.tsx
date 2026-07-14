"use client";

import { useRef } from "react";
import ImageCompare from "@/src/components/CompairImage";
import api from "@/lib/axios";
import { useState } from "react";

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [enhancedImage, setEnhancedImage] = useState("");
  const [originImage, setOriginImage] = useState("");

  const handleAddImage = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];

    if (!file) return;

    console.log("Selected file:", file);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model_name", "CIDNet")

    console.log("FormData:");
    for (const pair of formData.entries()) {
      console.log(pair[0], pair[1]);
    }

    try {
      const response = await api.post(
        "/api/v1/enhance/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        },
      );

      setOriginImage(URL.createObjectURL(file))
      setEnhancedImage(response.data.data.enhanced_image_base64);

      setEnhancedImage(response.data.data.enhanced_image_base64)
   
      console.log(response.data);
    } catch (error) {
      console.error(error);
    }

    e.target.value = "";
  };

  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden p-5">
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(135deg,#F8BBD9 0%,#FDD5B4 25%,#FFF2CC 50%,#E1F5FE 75%,#BBDEFB 100%)",
        }}
      />

      <div className="absolute top-[-150px] left-[-120px] h-96 w-96 rounded-full bg-pink-300/40 blur-3xl" />
      <div className="absolute bottom-[-150px] right-[-120px] h-96 w-96 rounded-full bg-sky-300/40 blur-3xl" />

      <section className="relative z-10 flex flex-col items-center gap-8">
        <h1 className="bg-gradient-to-r from-pink-500 via-orange-400 to-blue-500 bg-clip-text text-2xl font-extrabold text-transparent md:text-9xl">
          Lowlight Image Enhance
        </h1>

        <ImageCompare enhancedImage={enhancedImage} originImage={originImage}/>

        <div
          onClick={handleAddImage}
          className="cursor-pointer rounded-lg bg-black px-6 py-3 text-white hover:bg-gray-800"
        >
          Upload Image
        </div>

        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
      </section>
    </main>
  );
}