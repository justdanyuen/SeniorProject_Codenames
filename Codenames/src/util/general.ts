export function getRole(message: string | undefined): string[] | null {
  if (!message) return null;
  const startSymbol = "[";
  const endSymbol = "]";
  const start = message.indexOf(startSymbol);
  const end = message.indexOf(endSymbol);
  if (start != 0 || end == -1) return null;
  return [message.substring(start + 1, end), message.substring(end + 1)];
}

export function cardTypeToColor(type: string, guessed: boolean) {
  if (!guessed) {
    return "#fff";
  }
  switch (type) {
    case "Red":
      return "#a83232";
    case "Blue":
      return "#323ea8";
    case "Civilian":
      return "#a87d32";
    case "Assassin":
      return "#9132a8";
  }
}
