"use client";

import { Button } from "@heroui/button";
import { Image, Skeleton } from "@heroui/react";
import { FcAddImage } from "react-icons/fc";
import { useEffect, useRef, useState } from "react";

import { title, subtitle } from "@/components/primitives";
import PredictCard from "@/components/card";
import usePredictService from "@/service/predict_service";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const {
    prediction,
    loading,
    gradcamImage,
    handleFileChange,
    handleUpload,
    handleUploadGradcam,
  } = usePredictService(file, setFile);

  const handleButtonClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileInputChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    handleFileChange(event);
  };

  useEffect(() => {
    if (file) {
      handleUpload(file);
      handleUploadGradcam(file);
    }
  }, [file]);

  return (
    <section className="flex flex-col items-center justify-center gap-4">
      <div className="inline-block max-w-2xl text-center justify-center">
        <span className={title()}>AI-powered&nbsp;</span>
        <span className={title({ color: "violet" })}>Pneumonia&nbsp;</span>
        <br />
        <span className={title()}>Detection from Chest X-ray.</span>
        <div className={subtitle({ class: "mt-4" })}>
          Accurate, fast, and reliable deep learning-based diagnosis.
        </div>
      </div>

      <div>
        <PredictCard file={file} loading={loading} prediction={prediction} />
      </div>

      <div className="flex gap-3">
        <Button
          className="bg-gradient-to-tr from-pink-500 to-yellow-500 text-white shadow-lg"
          radius="full"
          size="lg"
          startContent={<FcAddImage size={25} />}
          onPress={handleButtonClick}
        >
          Upload your X-ray image
        </Button>
      </div>

      {/* field to upload image */}
      <input
        ref={fileInputRef}
        style={{ display: "none" }}
        type="file"
        onChange={handleFileInputChange}
      />
      {loading ? (
        <Skeleton className="rounded-lg" isLoaded={loading}>
          <Image
            alt="Woman listing to music"
            className="object-cover"
            src={
              gradcamImage
                ? `data:image/png;base64,${gradcamImage}`
                : "grad.png"
            }
            style={{ width: "1500px", height: "450px" }}
          />
        </Skeleton>
      ) : (
        <>
          <div className="inline-block max-w-2xl text-center justify-center">
            <span className={title()}>AI-powered&nbsp;</span>
            <span className={title({ color: "violet" })}>Pneumonia&nbsp;</span>
            <br />
            <span className={title()}>Detection with GradCAM.</span>
          </div>
          <Image
            alt="Woman listing to music"
            className="object-cover"
            src={
              gradcamImage
                ? `data:image/png;base64,${gradcamImage}`
                : "grad.png"
            }
            style={{ width: "1500px", height: "450px" }}
          />
        </>
      )}
      {/* <div className="mt-8">
        <Snippet hideCopyButton hideSymbol variant="bordered">
          <span>
            Get started by editing <Code color="primary">app/page.tsx</Code>
          </span>
        </Snippet>
      </div> */}
    </section>
  );
}
