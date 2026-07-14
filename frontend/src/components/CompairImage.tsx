"use client"

import { useState } from "react"

interface ImageCompareProps {
  enhancedImage: string;
  originImage: any
}

export default function ImageCompareImageCompare({enhancedImage, originImage}: ImageCompareProps) {
  const [value, setValue] = useState(50)

  return (
    <div className="relative w-full max-w-[650px] aspect-[4.5/3] overflow-hidden rounded-2xl">      
      <img src={originImage} className="absolute w-full h-full object-cover"/>

      <div
        className="absolute inset-0 overflow-hidden bg-white"
        style={{ clipPath: `inset(0 ${100 - value}% 0 0)`}}
      >
        <img
          src={enhancedImage}
          className="absolute insert-0 h-full object-cover"
        />
      </div>

      <input
        type="range"
        min="0"
        max="100"
        value={value}
        onChange={(e) => setValue(Number(e.target.value))}
        className="absolute bottom-2 left-1/2 top-1/4 -translate-x-1/2 w-[100%] opacity-0 w-full h-full"
      />
    </div>
  )
}