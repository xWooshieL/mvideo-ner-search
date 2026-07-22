; установщик М.Видео Разметка v0.1.0 — стиль как у ГК МОС
#define AppName "М.Видео Разметка"
#define AppVersion "0.1.0"
#define AppPublisher "Команда буткемпа М.Видео"
#define AppExeName "MvLabel.exe"
#define DistDir "C:\mv-app\dist\mvlabel"

[Setup]
AppId={{8C3F5D17-92A6-4B1E-B738-2D4E6F8A0C55}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=© 2026 Команда буткемпа М.Видео
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName}
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}.0
DefaultDirName={autopf}\МВидео Разметка
DefaultGroupName=М.Видео
UsePreviousAppDir=yes
DisableDirPage=no
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
OutputDir=output
OutputBaseFilename=MVideo_Labeling_Setup_{#AppVersion}
SetupIconFile=..\cpp\mvlabel\assets\icon.ico
WizardImageFile=assets\wizard-sidebar.bmp
WizardSmallImageFile=assets\wizard-header.bmp
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableProgramGroupPage=yes
DisableWelcomePage=no
ShowLanguageDialog=no
AllowNoIcons=yes
MinVersion=10.0
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=force

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Messages]
WelcomeLabel1=Добро пожаловать в установку %n
WelcomeLabel2=Программа установит [name/ver] на ваш компьютер.%n%nТрёхэтапный мастер разметки: B/I/O, тип сущности, подтип атрибута — плюс режим соответствия 1/0.%n%nНажмите «Далее», чтобы начать установку.

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительно:"; Flags: checkedonce

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pdb,mvlabel.ini"

[Icons]
Name: "{group}\{#AppName} — Никита"; Filename: "{app}\{#AppExeName}"; Parameters: "nikita"; Comment: "Разметка — Никита"
Name: "{group}\{#AppName} — Некит"; Filename: "{app}\{#AppExeName}"; Parameters: "nekit"; Comment: "Разметка — Некит"
Name: "{group}\{#AppName} — Лиза"; Filename: "{app}\{#AppExeName}"; Parameters: "liza"; Comment: "Разметка — Лиза"
Name: "{group}\Удалить {#AppName}"; Filename: "{uninstallexe}"; Comment: "Удалить программу"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "Разметка М.Видео"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser shellexec

[Code]
procedure KillAppProcess();
var
  ResultCode: Integer;
begin
  if Exec('taskkill', '/IM MvLabel.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    Sleep(400);
end;

function InitializeSetup(): Boolean;
begin
  KillAppProcess();
  Result := True;
end;

function InitializeUninstall(): Boolean;
begin
  KillAppProcess();
  Result := True;
end;

procedure InitializeWizard();
begin
  WizardForm.Color := clWhite;
  WizardForm.InnerPage.Color := clWhite;
  WizardForm.WelcomePage.Color := clWhite;
  WizardForm.FinishedPage.Color := clWhite;
  WizardForm.WelcomeLabel1.Font.Color := $001A1818;
  WizardForm.WelcomeLabel1.Font.Style := [fsBold];
  WizardForm.WelcomeLabel2.Font.Color := $0067647A;
  WizardForm.PageNameLabel.Font.Color := $001A1818;
  WizardForm.PageDescriptionLabel.Font.Color := $0067647A;
end;
