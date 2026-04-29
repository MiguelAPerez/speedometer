; installer.nsi — NSIS installer script for Speedometer (Windows)
; Run from the project root: makensis build/installer.nsi
; Requires: dist\Speedometer\ folder produced by PyInstaller

!define APP_NAME    "Speedometer"
!define COMPANY     "Speedometer"
!define APP_VERSION "$%APP_VERSION%"
!define OUT_FILE    "Speedometer-${APP_VERSION}-windows-setup.exe"

Name            "${APP_NAME} ${APP_VERSION}"
OutFile         "${OUT_FILE}"
InstallDir      "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKLM "Software\${COMPANY}\${APP_NAME}" "Install_Dir"
RequestExecutionLevel admin
SetCompressor    lzma

;----------------------------------------------------------------------
; Pages
;----------------------------------------------------------------------
Page directory
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

;----------------------------------------------------------------------
; Install
;----------------------------------------------------------------------
Section "Install"
  SetOutPath "$INSTDIR"
  File /r "..\dist\Speedometer\*"

  ; Registry entries for Add/Remove Programs
  WriteRegStr   HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "DisplayName"    "${APP_NAME}"
  WriteRegStr   HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "DisplayVersion" "${APP_VERSION}"
  WriteRegStr   HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "Publisher"      "${COMPANY}"
  WriteRegStr   HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "UninstallString" '"$INSTDIR\Uninstall.exe"'
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "NoModify" 1
  WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" \
                "NoRepair" 1

  WriteUninstaller "$INSTDIR\Uninstall.exe"

  ; Shortcuts
  CreateDirectory "$SMPROGRAMS\${APP_NAME}"
  CreateShortCut  "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"  "$INSTDIR\Speedometer.exe"
  CreateShortCut  "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"    "$INSTDIR\Uninstall.exe"
  CreateShortCut  "$DESKTOP\${APP_NAME}.lnk"                 "$INSTDIR\Speedometer.exe"
SectionEnd

;----------------------------------------------------------------------
; Uninstall
;----------------------------------------------------------------------
Section "Uninstall"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR"

  Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
  Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
  RMDir  "$SMPROGRAMS\${APP_NAME}"
  Delete "$DESKTOP\${APP_NAME}.lnk"

  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
  DeleteRegKey HKLM "Software\${COMPANY}\${APP_NAME}"
SectionEnd
