import { useState } from "react";

import { postformDataAxios } from "@/utils/axios";

interface PredictResponse {
  label: string;
  probability: number;
  message: string;
}

interface PredictGradcamResponse extends PredictResponse {
  gradcam_image: string;
}

interface PredictObjectResponse {
  result: PredictResponse;
}

interface PredictGradcamObjectResponse {
  result: PredictGradcamResponse;
}

const usePredictService = (
  file: File | null,
  setFile: React.Dispatch<React.SetStateAction<File | null>>,
) => {
  const [loading, setLoading] = useState(false);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [gradcamImage, setGradcamImage] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const validateImage = (file: File | null): boolean => {
    if (!file) {
      alert("Vui lòng chọn một ảnh!");

      return false;
    }
    if (!file.type.startsWith("image/")) {
      alert("Vui lòng chọn một tệp hình ảnh!");

      return false;
    }

    return true;
  };

  const handleUpload = async (file: File | null) => {
    if (!validateImage(file)) return;

    const formData = new FormData();

    formData.append("file", file!);

    setLoading(true);
    setGradcamImage(null); // reset nếu trước đó đã dùng GradCAM

    try {
      const response = await postformDataAxios<PredictObjectResponse>(
        `predict`,
        formData,
      );
      const result = response.data.result;

      setPrediction(
        `${result.message} Viêm phổi (Xác suất: ${(result.probability * 100).toFixed(2)}%)`,
      );
    } catch (error) {
      setPrediction(`Dự đoán thất bại: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  const handleUploadGradcam = async (file: File | null) => {
    if (!validateImage(file)) return;

    const formData = new FormData();

    formData.append("file", file!);

    setGradcamLoading(true);

    try {
      const response = await postformDataAxios<PredictGradcamObjectResponse>(
        `predict_gradcam`,
        formData,
      );

      const result = response.data.result;

      // setPrediction(
      //   `${result.message} (Xác suất: ${(result.probability * 100).toFixed(2)}%)`,
      // );

      setGradcamImage(result.gradcam_image);
    } catch (error) {
      setPrediction(`Dự đoán + GradCAM thất bại: ${error}`);
    } finally {
      setGradcamLoading(false);
    }
  };

  return {
    loading,
    gradcamLoading,
    prediction,
    gradcamImage,
    handleFileChange,
    handleUpload,
    handleUploadGradcam,
  };
};

export default usePredictService;
