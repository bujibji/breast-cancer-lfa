
{ pkgs }: {
  deps = [
    pkgs.libGL
    pkgs.glib
    pkgs.libxcrypt
    pkgs.zlib
    pkgs.glibcLocales
  ];
}
