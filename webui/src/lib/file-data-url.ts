/** Read a File as a ``data:<mime>;base64,...`` URL for gateway WebSocket uploads. */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("failed to read file"));
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== "string" || !result.startsWith("data:")) {
        reject(new Error("invalid file data URL"));
        return;
      }
      resolve(result);
    };
    reader.readAsDataURL(file);
  });
}
