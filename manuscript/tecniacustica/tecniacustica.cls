% =====================================================================
% tecniacustica.cls — Plantilla NO oficial que reproduce visualmente
% el formato de Tecniacústica (basada en el ejemplo publicado del 53º
% Congreso, Elche 2022, disponible en documentacion.sea-acustica.es).
%
% Si el comité del congreso publica una plantilla oficial, sustituirla
% por esta. Esta clase es un esqueleto razonable mientras tanto.
% =====================================================================

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tecniacustica}[2026/01/15 Tecniacústica look-alike]

\LoadClass[a4paper,11pt]{article}

\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage[spanish]{babel}
\RequirePackage[margin=2.5cm,top=3.3cm,bottom=2.5cm,headheight=2cm]{geometry}
\RequirePackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\RequirePackage{graphicx}
\RequirePackage{fancyhdr}
\RequirePackage{xcolor}
\RequirePackage{titlesec}
\RequirePackage{booktabs}
\RequirePackage{hyperref}
\hypersetup{hidelinks}
\RequirePackage{enumitem}
\RequirePackage{caption}
\captionsetup{font=small,labelfont=bf,labelsep=endash}

% Color verde Tecniacústica (aproximado al PDF de muestra Elche22)
\definecolor{tecnigreen}{HTML}{2C8C3D}

% Header con título del congreso (sustituir logo cuando haya uno oficial)
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{%
  % Para incluir el logo oficial, descargar de tecniacustica.es y
  % descomentar:
  % \includegraphics[height=1.7cm]{tecniacustica-logo.png}
  \color{tecnigreen}\bfseries\sffamily\large TECNI\textit{\textbf{ACUSTICA}}\\
  \color{tecnigreen}\sffamily\footnotesize GRANADA · 2026
}
\fancyhead[R]{%
  \color{tecnigreen}\sffamily\bfseries\large 57º CONGRESO ESPAÑOL DE ACÚSTICA\\
  \color{tecnigreen}\sffamily\small XIII CONGRESO IBÉRICO DE ACÚSTICA
}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Formato de secciones: numéricas, mayúsculas, bold
\titleformat{\section}{\normalfont\bfseries\sffamily}{\thesection.}{1ex}{\MakeUppercase}
\titleformat{\subsection}{\normalfont\bfseries\sffamily}{\thesubsection.}{1ex}{}
\titleformat{\subsubsection}{\normalfont\bfseries\sffamily\itshape}{\thesubsubsection.}{1ex}{}
\titlespacing*{\section}{0pt}{1.2em}{0.6em}
\titlespacing*{\subsection}{0pt}{1em}{0.4em}

% Comandos auxiliares: \pacs, \keywords, \abstractbox
\newcommand{\pacs}[1]{\noindent\textbf{PACS:} #1\par\smallskip}
\newcommand{\keywords}[1]{\noindent\textbf{Palabras clave:} #1\par\smallskip}
\newcommand{\engkeywords}[1]{\noindent\textbf{Keywords:} #1\par\smallskip}

% Bloque autores: simple, una linea por autor con afiliacion
% (anonimizar durante el doble ciego)
\newcommand{\authorsblock}[1]{%
  \par\noindent #1\par
}

% Título centrado en mayúsculas, bold
\renewcommand{\maketitle}{%
  \begin{center}
  \bfseries\sffamily\Large\MakeUppercase{\@title}
  \end{center}
  \vspace{0.5em}
}

% Resumen y abstract con encabezado bold sin numeración
\newenvironment{resumen}{\par\noindent\textbf{RESUMEN}\par\smallskip}{\par\smallskip}
\newenvironment{englishabstract}{\par\noindent\textbf{ABSTRACT}\par\smallskip}{\par\smallskip}

\endinput
