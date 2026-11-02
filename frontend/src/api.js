// Sends form fields to a backend endpoint and returns the JSON response.
export async function postForm(url, fields) {
  const body = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    body.append(key, value);
  }

  const response = await fetch(url, { method: "POST", body });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}
