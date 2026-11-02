// Returns the JSON response, or throws with the server's message ("detail") if there is one.
async function handle(response) {
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const data = await response.json();
      if (data.detail) message = data.detail;
    } catch {
      // not JSON: keep the status text
    }
    throw new Error(message);
  }
  return response.json();
}

// Sends form fields (strings or files) to a backend endpoint and returns the JSON response.
export async function postForm(url, fields) {
  const body = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    body.append(key, value);
  }
  return handle(await fetch(url, { method: "POST", body }));
}

export async function getJson(url, params = {}) {
  const query = new URLSearchParams(params).toString();
  return handle(await fetch(query ? `${url}?${query}` : url));
}
