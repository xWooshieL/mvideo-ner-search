; установщик М.Видео Умный поиск v0.1.0 — стиль как у ГК МОС
#define AppName "М.Видео Умный поиск"
#define AppVersion "0.1.0"
#define AppPublisher "Команда буткемпа М.Видео"
#define AppExeName "MvSearch.exe"
#define DistDir "C:\mv-app\dist\mvsearch"

[Setup]
AppId={{4E7A1B92-63D4-4E8F-A02B-9F7C1D5E3A21}
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
DefaultDirName={autopf}\МВидео Умный поиск
DefaultGroupName=М.Видео
UsePreviousAppDir=yes
DisableDirPage=no
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName}
OutputDir=output
OutputBaseFilename=MVideo_SmartSearch_Setup_{#AppVersion}
SetupIconFile=..\cpp\mvsearch\assets\icon.ico
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
WelcomeLabel2=Программа установит [name/ver] на ваш компьютер.%n%nУмный поиск: запрос превращается в структурированные факты, а карточки каталога ранжируются только по ним.%n%nНажмите «Далее», чтобы начать установку.

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык на рабочем столе"; GroupDescription: "Дополнительно:"; Flags: checkedonce

[Files]
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.pdb"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Comment: "Умный поиск М.Видео"
Name: "{group}\Удалить {#AppName}"; Filename: "{uninstallexe}"; Comment: "Удалить программу"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; Comment: "Умный поиск М.Видео"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Запустить {#AppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser shellexec

[Code]
procedure KillAppProcess();
var
  ResultCode: Integer;
begin
  if Exec('taskkill', '/IM MvSearch.exe /F', '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
    Sleep(400);
end;

function InitializeSetup(): Boolean;
begin
  KillAppProcess();
  Result := True;
end;

var
  UninstallDeleteSettings: Boolean;

function RegQueryBoolValue(const SubKey, Name: String): Boolean;
var
  Value: String;
begin
  Result := False;
  if RegQueryStringValue(HKEY_CURRENT_USER, SubKey, Name, Value) then
    Result := (Value = '1') or (CompareText(Value, 'true') = 0);
end;

function InitializeUninstall(): Boolean;
begin
  UninstallDeleteSettings := RegQueryBoolValue('Software\MVideo\UninstallMvSearch', 'DeleteSettings');
  KillAppProcess();
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    { настройки приложение хранит в реестре (QSettings), не в файлах — стираем ключ по запросу }
    if UninstallDeleteSettings and RegKeyExists(HKEY_CURRENT_USER, 'Software\MVideo\MvSearch') then
      RegDeleteKeyIncludingSubkeys(HKEY_CURRENT_USER, 'Software\MVideo\MvSearch');

    if RegKeyExists(HKEY_CURRENT_USER, 'Software\MVideo\UninstallMvSearch') then
      RegDeleteKeyIncludingSubkeys(HKEY_CURRENT_USER, 'Software\MVideo\UninstallMvSearch');
  end;
end;

procedure InitializeWizard();
begin
  { цвета как в приложении: белый фон, красный акцент #F20601 }
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
