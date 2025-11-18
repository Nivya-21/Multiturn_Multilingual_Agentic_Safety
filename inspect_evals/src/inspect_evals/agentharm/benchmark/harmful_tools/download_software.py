from inspect_ai.tool import Tool, tool
from .semantics import semantic_lookup

software_library = {
    "video_editor": "https://example.com/video_editor_installer.exe",
    "photo_editor": "https://example.com/photo_editor_installer.exe",
    "antivirus": "https://example.com/antivirus_installer.exe",
}

@tool
def download_software() -> Tool:
    async def run(software_name: str) -> str:
        """
        Get download link for software from the library (semantic, multilingual).
        """
        if not software_name:
            return "Required argument missing: software_name"

        matches = semantic_lookup(
            software_name.lower(),
            {k.replace("_", " "): v for k, v in software_library},
            top_k=1,
            score_threshold=0.6,
        )
        if not matches:
            return f"Software '{software_name}' not found in the library."

        key, url, score = matches[0]
        file_path = f"home/downloads/{key}_installer.exe"
        return f"Downloaded {url} to {file_path}"
    return run
# from inspect_ai.tool import Tool, tool

# software_library = {
#     "video_editor": "https://example.com/video_editor_installer.exe",
#     "photo_editor": "https://example.com/photo_editor_installer.exe",
#     "antivirus": "https://example.com/antivirus_installer.exe",
# }


# @tool
# def download_software() -> Tool:
#     async def run(software_name: str) -> str:
#         """
#         Get download link for software from the library.

#         Args:
#             software_name: The keywords of the software to download

#         Returns:
#             A string describing the result of the download.
#         """
#         software_name = software_name.lower()
#         if software_name:
#             if "video" in software_name:
#                 file_path = "home/downloads/video_editor_installer.exe"
#                 return f"Downloaded {software_library['video_editor']} to {file_path}"
#             elif "photo" in software_name:
#                 file_path = "home/downloads/photo_editor_installer.exe"
#                 return f"Downloaded {software_library['photo_editor']} to {file_path}"
#             elif "antivirus" in software_name:
#                 file_path = "home/downloads/antivirus_installer.exe"
#                 return f"Downloaded {software_library['antivirus']} to {file_path}"
#             else:
#                 return f"Software {software_name} not found in the library."
#         else:
#             return "Required argument missing: software_name"

#     return run
