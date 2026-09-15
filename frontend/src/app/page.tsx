"use client"

import { useEffect, useState, type ReactNode } from "react"

import { BlurFade } from "@/components/ui/blur-fade"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { GridPattern } from "@/components/ui/grid-pattern"
import { NumberTicker } from "@/components/ui/number-ticker"
import { RippleButton } from "@/components/ui/ripple-button"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Textarea } from "@/components/ui/textarea"
import {
  ApiError,
  matchResume,
  type KeywordWeight,
  type MatchResponse,
  type OccupationMatch,
} from "@/lib/api"
import { EXAMPLE_JOB_DESCRIPTION, EXAMPLE_RESUME } from "@/lib/example-data"
import { cn } from "@/lib/utils"

const MIN_LENGTH = 50

type Status = "idle" | "loading" | "success" | "error"

interface ErrorInfo {
  status: number
  message: string
}

export default function Home() {
  const [jobDescription, setJobDescription] = useState("")
  const [resume, setResume] = useState("")
  const [status, setStatus] = useState<Status>("idle")
  const [result, setResult] = useState<MatchResponse | null>(null)
  const [error, setError] = useState<ErrorInfo | null>(null)
  const [showColdStartHint, setShowColdStartHint] = useState(false)

  const jobValid = jobDescription.length >= MIN_LENGTH
  const resumeValid = resume.length >= MIN_LENGTH
  const canSubmit = jobValid && resumeValid && status !== "loading"

  useEffect(() => {
    if (status !== "loading") return
    const timer = setTimeout(() => setShowColdStartHint(true), 4000)
    return () => {
      clearTimeout(timer)
      setShowColdStartHint(false)
    }
  }, [status])

  async function handleAnalyze() {
    if (!jobValid || !resumeValid) return
    setStatus("loading")
    setError(null)
    try {
      const data = await matchResume({ jobDescription, resume, topK: 5 })
      setResult(data)
      setStatus("success")
    } catch (err) {
      if (err instanceof ApiError) {
        setError({ status: err.status, message: err.message })
      } else {
        setError({
          status: 0,
          message:
            err instanceof Error
              ? err.message
              : "Could not reach the API. Check your connection and try again.",
        })
      }
      setStatus("error")
    }
  }

  function handleLoadExample() {
    setJobDescription(EXAMPLE_JOB_DESCRIPTION)
    setResume(EXAMPLE_RESUME)
  }

  return (
    <div className="relative min-h-screen">
      <GridPattern
        className={cn(
          "absolute inset-0 -z-10 h-full w-full",
          "fill-foreground/[0.015] stroke-foreground/[0.03]",
          "[mask-image:radial-gradient(ellipse_65%_55%_at_50%_0%,black,transparent)]"
        )}
      />

      <div className="relative z-10 mx-auto flex max-w-4xl flex-col gap-12 px-4 pt-16 pb-24 sm:px-6">
        <header className="flex flex-col items-center gap-3 text-center">
          <h1 className="font-display text-4xl font-medium tracking-tight text-foreground sm:text-5xl">
            Resume Matcher
          </h1>
          <p className="max-w-xl text-sm text-balance text-muted-foreground sm:text-base">
            Paste a job posting and your resume. See exactly which keywords
            the posting wants that your resume is missing, scored against
            real O*NET occupation data.
          </p>
        </header>

        <section className="flex flex-col gap-4">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <TextField
              label="Job posting"
              placeholder="Paste the job description here..."
              value={jobDescription}
              onChange={setJobDescription}
              valid={jobValid}
            />
            <TextField
              label="Your resume"
              placeholder="Paste your resume here..."
              value={resume}
              onChange={setResume}
              valid={resumeValid}
            />
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <RippleButton
              onClick={handleAnalyze}
              disabled={!canSubmit}
              rippleColor="rgba(255,255,255,0.35)"
              className={cn(
                "px-6 py-2.5 text-sm font-medium transition-colors",
                canSubmit
                  ? "border-primary bg-primary text-primary-foreground hover:bg-primary/90"
                  : "cursor-not-allowed border-transparent bg-muted text-muted-foreground"
              )}
            >
              {status === "loading" ? "Analyzing..." : "Analyze"}
            </RippleButton>
            <Button
              type="button"
              variant="outline"
              onClick={handleLoadExample}
              disabled={status === "loading"}
            >
              Load example
            </Button>
            {(!jobValid || !resumeValid) && (
              <span className="text-xs text-muted-foreground">
                Both fields need at least {MIN_LENGTH} characters.
              </span>
            )}
          </div>
        </section>

        <Separator />

        <section aria-live="polite" className="flex flex-col gap-10">
          {status === "idle" && <EmptyState />}
          {status === "loading" && (
            <ResultsSkeleton showColdStartHint={showColdStartHint} />
          )}
          {status === "error" && error && <ErrorState error={error} />}
          {status === "success" && result && <Results result={result} />}
        </section>
      </div>
    </div>
  )
}

function TextField({
  label,
  placeholder,
  value,
  onChange,
  valid,
}: {
  label: string
  placeholder: string
  value: string
  onChange: (value: string) => void
  valid: boolean
}) {
  return (
    <Card className="gap-3 py-4">
      <CardHeader className="flex flex-row items-center justify-between gap-2 px-4">
        <CardTitle className="text-sm font-medium">{label}</CardTitle>
        <span
          className={cn(
            "text-xs tabular-nums",
            valid ? "text-emerald-400" : "text-muted-foreground"
          )}
        >
          {value.length} / {MIN_LENGTH}+
        </span>
      </CardHeader>
      <CardContent className="px-4">
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="min-h-40 resize-y text-sm"
        />
      </CardContent>
    </Card>
  )
}

function EmptyState() {
  return (
    <div className="rounded-lg border border-dashed border-border py-12 text-center text-sm text-muted-foreground">
      Run an analysis to see missing keywords, a match score, and the
      closest O*NET occupation matches here.
    </div>
  )
}

function ErrorState({ error }: { error: ErrorInfo }) {
  const { title, hint } = (() => {
    if (error.status === 422) {
      return {
        title: "Invalid input",
        hint: "The request didn't meet the API's validation rules.",
      }
    }
    if (error.status === 503) {
      return {
        title: "Model not ready",
        hint: "The API is up, but the model artifact isn't loaded.",
      }
    }
    return {
      title: error.status
        ? `Request failed (${error.status})`
        : "Request failed",
      hint: "Something went wrong reaching the API.",
    }
  })()

  return (
    <Card className="border-destructive/40 bg-destructive/5">
      <CardHeader>
        <CardTitle className="text-base text-destructive">{title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-1 text-sm">
        <p className="text-muted-foreground">{hint}</p>
        <p className="font-mono text-xs break-words text-destructive">
          {error.message}
        </p>
      </CardContent>
    </Card>
  )
}

function ResultsSkeleton({
  showColdStartHint,
}: {
  showColdStartHint: boolean
}) {
  return (
    <div className="flex flex-col gap-10">
      <div className="flex flex-col gap-3">
        <Skeleton className="h-5 w-72 max-w-full" />
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-7 w-24 rounded-full" />
          ))}
        </div>
      </div>
      <Skeleton className="h-14 w-44" />
      <div className="flex flex-col gap-3">
        <Skeleton className="h-4 w-40" />
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-6 w-20 rounded-full" />
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <Skeleton className="h-4 w-56" />
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-10 w-full" />
        ))}
      </div>
      {showColdStartHint && (
        <p className="text-center text-xs text-muted-foreground">
          Waking up the model — first request after idle takes a few
          seconds.
        </p>
      )}
    </div>
  )
}

function Results({ result }: { result: MatchResponse }) {
  return (
    <div className="flex flex-col gap-12">
      <MissingKeywords keywords={result.missing_keywords} />
      <MatchScore score={result.match_score} />
      <MatchedKeywords keywords={result.matched_keywords} />
      <ClosestOccupations occupations={result.closest_occupations} />
    </div>
  )
}

function MissingKeywords({ keywords }: { keywords: KeywordWeight[] }) {
  const maxWeight = Math.max(0, ...keywords.map((k) => k.weight))

  return (
    <div className="flex flex-col gap-4">
      <SectionHeading>
        This posting wants, and your resume doesn&apos;t mention
      </SectionHeading>
      {keywords.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Nothing stood out — your resume already covers this posting&apos;s
          key terms.
        </p>
      ) : (
        <div className="flex flex-wrap gap-2.5">
          {keywords.map((k, i) => {
            const t = maxWeight > 0 ? k.weight / maxWeight : 0
            return (
              <BlurFade key={k.term} delay={i * 0.04} direction="up" offset={8}>
                <span
                  className="inline-flex items-center rounded-full border border-primary/30 bg-primary/10 px-3.5 py-1.5 font-medium text-foreground"
                  style={{
                    fontSize: `${0.78 + t * 0.42}rem`,
                    opacity: 0.65 + t * 0.35,
                  }}
                >
                  {k.term}
                </span>
              </BlurFade>
            )
          })}
        </div>
      )}
    </div>
  )
}

function MatchScore({ score }: { score: number }) {
  return (
    <div className="flex w-fit items-center gap-4 rounded-lg border border-border/60 bg-card/50 px-5 py-4">
      <span className="text-xs tracking-wide text-muted-foreground uppercase">
        Match score
      </span>
      <span className="flex items-baseline gap-0.5 text-2xl font-semibold">
        <NumberTicker
          value={score}
          decimalPlaces={1}
          className="text-foreground"
        />
        <span className="text-foreground">%</span>
      </span>
    </div>
  )
}

function MatchedKeywords({ keywords }: { keywords: KeywordWeight[] }) {
  if (keywords.length === 0) return null

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
        Already covered
      </h3>
      <div className="flex flex-wrap gap-2">
        {keywords.map((k, i) => (
          <BlurFade key={k.term} delay={i * 0.03} direction="up" offset={6}>
            <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground">
              {k.term}
            </span>
          </BlurFade>
        ))}
      </div>
    </div>
  )
}

function ClosestOccupations({
  occupations,
}: {
  occupations: OccupationMatch[]
}) {
  return (
    <div className="flex flex-col gap-3">
      <SectionHeading small>Closest occupation matches</SectionHeading>
      <p className="-mt-2 text-xs text-muted-foreground">
        A similarity ranking against O*NET occupation text, based on your
        resume alone — not an occupation assignment.
      </p>
      <div className="flex flex-col divide-y divide-border/60 rounded-lg border border-border/60">
        {occupations.map((occ, i) => (
          <BlurFade key={occ.soc_code} delay={i * 0.04} direction="up" offset={6}>
            <div className="flex items-center justify-between gap-3 px-4 py-3 text-sm">
              <span className="min-w-0 flex-1 text-foreground">{occ.title}</span>
              <span className="shrink-0 tabular-nums text-muted-foreground">
                {(occ.score * 100).toFixed(1)}%
              </span>
            </div>
          </BlurFade>
        ))}
      </div>
    </div>
  )
}

function SectionHeading({
  children,
  small,
}: {
  children: ReactNode
  small?: boolean
}) {
  return (
    <h2
      className={cn(
        "font-semibold tracking-tight",
        small ? "text-base" : "text-xl sm:text-2xl"
      )}
    >
      {children}
    </h2>
  )
}
