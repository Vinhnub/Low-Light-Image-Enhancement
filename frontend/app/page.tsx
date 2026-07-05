"use client"
import Image from "next/image";
import ImageCompare from "@/src/components/CompairImage.tsx";

export default function Home() {
  return (
  <div className="min-h-screen w-full relative flex flex-col flex-1 items-center justify-center">
    <div
      className="absolute inset-0 z-0"
      style={{
        background: `linear-gradient(135deg, #F8BBD9 0%, #FDD5B4 25%, #FFF2CC 50%, #E1F5FE 75%, #BBDEFB 100%)`,
      }}
    />
      <ImageCompare />
  </div>
  );
}