export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "https://andyh06--resume-matcher-fastapi-app.modal.run"

export interface KeywordWeight {
  term: string
  weight: number
}

export interface OccupationMatch {
  soc_code: string
  title: string
  score: number
}

export interface MatchResponse {
  match_score: number
  missing_keywords: KeywordWeight[]
  matched_keywords: KeywordWeight[]
  closest_occupations: OccupationMatch[]
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

interface ValidationDetailItem {
  loc?: Array<string | number>
  msg?: string
}

async function extractErrorMessage(response: Response): Promise<string> {
  let data: unknown
  try {
    data = await response.json()
  } catch {
    return response.statusText || `Request failed with status ${response.status}`
  }

  if (data && typeof data === "object" && "detail" in data) {
    const detail = (data as { detail: unknown }).detail

    if (typeof detail === "string") {
      return detail
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const d = item as ValidationDetailItem
          const field =
            Array.isArray(d.loc) && d.loc.length > 0
              ? d.loc[d.loc.length - 1]
              : "field"
          return `${field}: ${d.msg ?? "invalid value"}`
        })
        .join("; ")
    }
  }

  return response.statusText || `Request failed with status ${response.status}`
}

export async function matchResume(params: {
  jobDescription: string
  resume: string
  topK?: number
}): Promise<MatchResponse> {
  const response = await fetch(`${API_BASE_URL}/match`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      job_description: params.jobDescription,
      resume: params.resume,
      top_k: params.topK ?? 5,
    }),
  })

  if (!response.ok) {
    const message = await extractErrorMessage(response)
    throw new ApiError(response.status, message)
  }

  return (await response.json()) as MatchResponse
}
