"""Synthetic data generator for web-focused coding tasks.

Generates diverse training examples using template-based expansion with
framework-specific patterns, code variations, and realistic task descriptions.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PromptTemplate:
    instruction: str
    context: dict = field(default_factory=dict)
    output_template: str = ""
    domain: str = "general"
    difficulty: str = "easy"
    tags: list[str] = field(default_factory=list)


NEXTJS_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create a Next.js App Router page at /blog that fetches posts from Prisma and renders them server-side",
        context={
            "schema": "model Post { id Int @id @default(autoincrement()) title String content String createdAt DateTime @default(now()) }",
            "constraints": ["Use App Router", "TypeScript strict", "Server Component"],
        },
        output_template="""// app/blog/page.tsx
import { prisma } from '@/lib/db'
import Link from 'next/link'

export default async function BlogPage() {
  const posts = await prisma.post.findMany({
    orderBy: { createdAt: 'desc' },
  })
  return (
    <main>
      <h1>Blog</h1>
      {posts.map(post => (
        <article key={post.id}>
          <Link href={`/blog/${encodeURIComponent(post.title)}`}>
            <h2>{post.title}</h2>
          </Link>
          <p>{post.content.slice(0, 200)}...</p>
        </article>
      ))}
    </main>
  )
}""",
        domain="nextjs",
        difficulty="easy",
        tags=["app-router", "server-component", "prisma", "ssr"],
    ),
    PromptTemplate(
        instruction="Write a TypeScript utility type that converts a Prisma model's select type into a typed DTO",
        context={
            "constraints": ["TypeScript strict", "Generic utility"],
        },
        output_template="""type PrismaSelect<T> = T extends { select?: infer S }
  ? { [K in keyof S]: S[K] extends true ? (K extends keyof T ? T[K] : never) : never }
  : T

type PostDTO = PrismaSelect<Parameters<typeof prisma.post.findFirst>[0]>

const example: PostDTO = { id: 1, title: 'Hello' }""",
        domain="typescript",
        difficulty="intermediate",
        tags=["generics", "prisma", "types"],
    ),
    PromptTemplate(
        instruction="Create a Prisma schema for a blog with users, posts, comments, and tags (many-to-many)",
        context={
            "constraints": [
                "PostgreSQL",
                "Foreign keys",
                "Indexes on common query fields",
            ],
        },
        output_template="""model User {
  id        Int       @id @default(autoincrement())
  email     String    @unique
  name      String?
  posts     Post[]
  comments  Comment[]
  createdAt DateTime  @default(now())
}

model Post {
  id        Int       @id @default(autoincrement())
  title     String
  content   String
  published Boolean   @default(false)
  author    User      @relation(fields: [authorId], references: [id])
  authorId  Int
  comments  Comment[]
  tags      PostTag[]
  createdAt DateTime  @default(now())

  @@index([authorId])
  @@index([published])
}

model Comment {
  id        Int      @id @default(autoincrement())
  body      String
  post      Post     @relation(fields: [postId], references: [id])
  postId    Int
  author    User     @relation(fields: [authorId], references: [id])
  authorId  Int
  createdAt DateTime @default(now())

  @@index([postId])
}

model Tag {
  id    Int       @id @default(autoincrement())
  name  String    @unique
  posts PostTag[]
}

model PostTag {
  post   Post @relation(fields: [postId], references: [id])
  postId Int
  tag    Tag  @relation(fields: [tagId], references: [id])
  tagId  Int

  @@id([postId, tagId])
}""",
        domain="prisma",
        difficulty="easy",
        tags=["schema", "relations", "many-to-many", "indexes"],
    ),
    PromptTemplate(
        instruction="Create a server action that creates a new comment and revalidates the post page",
        context={
            "constraints": ["Next.js Server Action", "TypeScript", "Zod validation"],
        },
        output_template="""'use server'

import { z } from 'zod'
import { revalidatePath } from 'next/cache'
import { prisma } from '@/lib/db'

const CommentSchema = z.object({
  postId: z.number().int().positive(),
  authorId: z.number().int().positive(),
  body: z.string().min(1).max(1000),
})

export async function createComment(formData: FormData) {
  const data = CommentSchema.parse({
    postId: Number(formData.get('postId')),
    authorId: Number(formData.get('authorId')),
    body: formData.get('body'),
  })

  await prisma.comment.create({ data })
  revalidatePath(`/blog/${data.postId}`)
}""",
        domain="nextjs",
        difficulty="intermediate",
        tags=["server-actions", "zod", "prisma", "revalidation"],
    ),
    PromptTemplate(
        instruction="Write a React client component with a loading state that displays a list of posts from a server component prop",
        context={
            "constraints": [
                "TypeScript",
                "Client component",
                "Loading/Empty/Error states",
            ],
        },
        output_template="""'use client'

import { useState } from 'react'

interface Post {
  id: number
  title: string
  content: string
}

interface PostListProps {
  posts: Promise<Post[]>
}

export default function PostList({ posts }: PostListProps) {
  const [resolvedPosts, setPosts] = useState<Post[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  posts.then(setPosts).catch(err => setError(err.message))

  if (error) {
    return <div className="text-red-500">Error: {error}</div>
  }

  if (resolvedPosts === null) {
    return <div className="animate-pulse">Loading posts...</div>
  }

  if (resolvedPosts.length === 0) {
    return <p>No posts yet.</p>
  }

  return (
    <ul>
      {resolvedPosts.map(post => (
        <li key={post.id}>
          <h3>{post.title}</h3>
          <p>{post.content}</p>
        </li>
      ))}
    </ul>
  )
}""",
        domain="react",
        difficulty="intermediate",
        tags=["client-component", "useState", "loading", "error-boundary"],
    ),
    PromptTemplate(
        instruction="Create a Prisma query with pagination, filtering, and sorting for a blog API route handler",
        context={
            "constraints": [
                "TypeScript",
                "Cursor pagination",
                "Multiple filter options",
            ],
        },
        output_template="""import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/db'

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const cursor = searchParams.get('cursor') ? Number(searchParams.get('cursor')) : undefined
  const limit = Math.min(Number(searchParams.get('limit')) || 10, 100)
  const search = searchParams.get('search') || ''
  const published = searchParams.get('published')

  const where = {
    ...(search ? {
      OR: [
        { title: { contains: search } },
        { content: { contains: search } },
      ],
    } : {}),
    ...(published !== null ? { published: published === 'true' } : {}),
  }

  const posts = await prisma.post.findMany({
    where,
    take: limit + 1,
    ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
    orderBy: { createdAt: 'desc' },
    include: {
      author: { select: { name: true } },
      _count: { select: { comments: true } },
    },
  })

  const hasMore = posts.length > limit
  const results = hasMore ? posts.slice(0, limit) : posts
  const nextCursor = hasMore ? results[results.length - 1].id : null

  return NextResponse.json({
    posts: results,
    nextCursor,
  })
}""",
        domain="nextjs",
        difficulty="intermediate",
        tags=["api-routes", "prisma", "pagination", "filtering"],
    ),
    PromptTemplate(
        instruction="Create a custom React hook for debounced search input",
        context={
            "constraints": ["TypeScript", "Generic hook", "AbortController support"],
        },
        output_template="""import { useState, useEffect, useRef } from 'react'

export function useDebouncedSearch<T>(
  searchFn: (query: string, signal: AbortSignal) => Promise<T[]>,
  delay: number = 300,
) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<T[]>([])
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    if (!query.trim()) {
      setResults([])
      return
    }

    const timer = setTimeout(async () => {
      setLoading(true)
      try {
        const data = await searchFn(query, controller.signal)
        setResults(data)
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          console.error(err)
        }
      } finally {
        setLoading(false)
      }
    }, delay)

    return () => clearTimeout(timer)
  }, [query, delay])

  return { query, setQuery, results, loading }
}""",
        domain="react",
        difficulty="intermediate",
        tags=["hooks", "debounce", "abort-controller", "generics"],
    ),
    PromptTemplate(
        instruction="Create a Next.js middleware that checks for authentication and redirects unauthenticated users",
        context={
            "constraints": [
                "Next.js Edge Middleware",
                "JWT via cookies",
                "Exclude public routes",
            ],
        },
        output_template="""import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

const PUBLIC_PATHS = ['/login', '/signup', '/api/auth']

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  if (PUBLIC_PATHS.some(path => pathname.startsWith(path))) {
    return NextResponse.next()
  }

  const token = request.cookies.get('auth-token')?.value

  if (!token) {
    const loginUrl = new URL('/login', request.url)
    loginUrl.searchParams.set('from', pathname)
    return NextResponse.redirect(loginUrl)
  }

  const response = NextResponse.next()
  response.headers.set('x-user-id', decodeToken(token).userId)
  return response
}

function decodeToken(token: string): { userId: string } {
  return JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString())
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
}""",
        domain="nextjs",
        difficulty="intermediate",
        tags=["middleware", "auth", "jwt", "edge"],
    ),
    PromptTemplate(
        instruction="Create a Prisma transaction that atomically transfers points between two users and creates an audit log",
        context={
            "constraints": ["Prisma transaction", "Atomic operations", "Audit trail"],
        },
        output_template="""import { prisma } from '@/lib/db'

export async function transferPoints(
  fromUserId: number,
  toUserId: number,
  amount: number,
) {
  return prisma.$transaction(async (tx) => {
    const sender = await tx.user.update({
      where: { id: fromUserId },
      data: { points: { decrement: amount } },
    })

    if (sender.points < 0) {
      throw new Error('Insufficient points')
    }

    const receiver = await tx.user.update({
      where: { id: toUserId },
      data: { points: { increment: amount } },
    })

    const audit = await tx.auditLog.create({
      data: {
        action: 'POINTS_TRANSFER',
        fromUserId,
        toUserId,
        amount,
        timestamp: new Date(),
      },
    })

    return { sender, receiver, audit }
  })
}""",
        domain="prisma",
        difficulty="intermediate",
        tags=["transactions", "atomic", "audit", "data-integrity"],
    ),
    PromptTemplate(
        instruction="Create a TypeScript Zod validation schema for a blog post creation API",
        context={
            "constraints": ["Zod", "TypeScript strict", "Detailed error messages"],
        },
        output_template="""import { z } from 'zod'

export const CreatePostSchema = z.object({
  title: z
    .string()
    .min(3, 'Title must be at least 3 characters')
    .max(200, 'Title must be under 200 characters'),
  content: z
    .string()
    .min(10, 'Content must be at least 10 characters'),
  published: z.boolean().default(false),
  tags: z
    .array(z.string().min(1).max(30))
    .max(10, 'Maximum 10 tags allowed')
    .default([]),
  authorId: z.number().int().positive('Invalid author ID'),
})

export type CreatePostInput = z.infer<typeof CreatePostSchema>""",
        domain="typescript",
        difficulty="easy",
        tags=["zod", "validation", "schema", "types"],
    ),
    PromptTemplate(
        instruction="Create an error boundary component that catches both render errors and async errors in Next.js",
        context={
            "constraints": [
                "React ErrorBoundary",
                "TypeScript",
                "Recovery action support",
            ],
        },
        output_template="""'use client'

import { Component, ReactNode } from 'react'
import { useEffect } from 'react'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onReset?: () => void
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

export default class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = { hasError: false, error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  reset = () => {
    this.setState({ hasError: false, error: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="border border-red-200 bg-red-50 p-4 rounded">
          <h2 className="text-red-800 font-bold">Something went wrong</h2>
          <p className="text-red-600 text-sm mt-1">
            {this.state.error?.message || 'An unexpected error occurred'}
          </p>
          <button
            onClick={this.reset}
            className="mt-3 px-4 py-2 bg-red-600 text-white rounded"
          >
            Try again
          </button>
        </div>
      )
    }

    return this.props.children
  }
}""",
        domain="react",
        difficulty="easy",
        tags=["error-boundary", "class-component", "error-handling"],
    ),
    PromptTemplate(
        instruction="Create a Next.js Route Handler for user registration with password hashing and duplicate email check",
        context={
            "constraints": [
                "Next.js Route Handler",
                "bcrypt",
                "Prisma",
                "Proper error responses",
            ],
        },
        output_template="""import { NextRequest, NextResponse } from 'next/server'
import { prisma } from '@/lib/db'
import bcrypt from 'bcryptjs'
import { CreateUserSchema } from '@/schemas/user'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const data = CreateUserSchema.parse(body)

    const existing = await prisma.user.findUnique({
      where: { email: data.email },
    })

    if (existing) {
      return NextResponse.json(
        { error: 'Email already registered' },
        { status: 409 },
      )
    }

    const hashedPassword = await bcrypt.hash(data.password, 12)

    const user = await prisma.user.create({
      data: {
        email: data.email,
        name: data.name,
        password: hashedPassword,
      },
      select: {
        id: true,
        email: true,
        name: true,
        createdAt: true,
      },
    })

    return NextResponse.json(user, { status: 201 })
  } catch (error) {
    if (error instanceof Error && error.name === 'ZodError') {
      return NextResponse.json(
        { error: 'Validation failed', details: JSON.parse(error.message) },
        { status: 400 },
      )
    }
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 },
    )
  }
}""",
        domain="nextjs",
        difficulty="intermediate",
        tags=["route-handler", "auth", "prisma", "validation"],
    ),
    PromptTemplate(
        instruction="Optimize this Prisma query to use proper includes and selects to avoid over-fetching",
        context={
            "problem": "const posts = await prisma.post.findMany({ include: { author: true, comments: true } })",
            "constraints": ["Prisma", "Minimize data transfer", "Keep type safety"],
        },
        output_template="""// Before (over-fetching):
// const posts = await prisma.post.findMany({
//   include: { author: true, comments: true }
// })

// After (optimized):
const posts = await prisma.post.findMany({
  select: {
    id: true,
    title: true,
    createdAt: true,
    author: {
      select: { id: true, name: true },
    },
    _count: {
      select: { comments: true },
    },
  },
  take: 20,
  orderBy: { createdAt: 'desc' },
})

// Type is now:
// { id: number; title: string; createdAt: Date;
//   author: { id: number; name: string | null };
//   _count: { comments: number } }[]""",
        domain="prisma",
        difficulty="easy",
        tags=["optimization", "select", "n+1", "performance"],
    ),
    PromptTemplate(
        instruction="Write a utility for generating typed Next.js route helpers with full type safety",
        context={
            "constraints": ["TypeScript", "Type-safe params", "Search params typing"],
        },
        output_template="""type RouteParams = Record<string, string | string[]>

export function typedRoute<T extends string>(path: T) {
  return {
    build: (params: Record<string, string | number> = {}) => {
      let result: string = path
      for (const [key, value] of Object.entries(params)) {
        result = result.replace(`[${key}]`, encodeURIComponent(String(value)))
      }
      return result
    },
    url: (params: Record<string, string | number> = {}) =>
      new URL(typedRoute(path).build(params), process.env.NEXT_PUBLIC_APP_URL!).toString(),
  }
}

const Routes = {
  home: () => '/',
  blog: {
    list: () => '/blog',
    post: (slug: string) => typedRoute('/blog/[slug]').build({ slug }),
    edit: (id: number) => typedRoute('/blog/[id]/edit').build({ id }),
  },
  api: {
    posts: () => '/api/posts',
    post: (id: number) => typedRoute('/api/posts/[id]').build({ id }),
  },
}

Routes.blog.post('my-first-post')
Routes.blog.edit(42)""",
        domain="typescript",
        difficulty="intermediate",
        tags=["routes", "type-safety", "utilities"],
    ),
    PromptTemplate(
        instruction="Create a Next.js layout component with a shared header and footer that highlights the active nav link",
        context={
            "constraints": ["App Router layout", "TypeScript", "Active link detection"],
        },
        output_template="""import Link from 'next/link'
import { ReactNode } from 'react'

interface LayoutProps {
  children: ReactNode
}

export default function BlogLayout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1 container mx-auto px-4 py-8">
        {children}
      </main>
      <Footer />
    </div>
  )
}

function Header() {
  return (
    <header className="border-b bg-white sticky top-0 z-50">
      <nav className="container mx-auto px-4 h-16 flex items-center gap-6">
        <Link href="/" className="font-bold text-xl">
          MyBlog
        </Link>
        <NavLink href="/blog">Blog</NavLink>
        <NavLink href="/about">About</NavLink>
      </nav>
    </header>
  )
}

function Footer() {
  return (
    <footer className="border-t py-8 mt-auto">
      <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
        &copy; {new Date().getFullYear()} MyBlog. All rights reserved.
      </div>
    </footer>
  )
}""",
        domain="nextjs",
        difficulty="easy",
        tags=["layout", "app-router", "navigation"],
    ),
    PromptTemplate(
        instruction="Create a Prisma seed script that populates a development database with realistic test data",
        context={
            "constraints": ["TypeScript", "Prisma", "faker-like data", "Idempotent"],
        },
        output_template="""import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

async function main() {
  await prisma.postTag.deleteMany()
  await prisma.comment.deleteMany()
  await prisma.post.deleteMany()
  await prisma.tag.deleteMany()
  await prisma.user.deleteMany()

  const users = await Promise.all(
    [...Array(10)].map((_, i) =>
      prisma.user.create({
        data: {
          email: `user${i + 1}@example.com`,
          name: `User ${i + 1}`,
        },
      })
    )
  )

  const tags = await Promise.all(
    ['typescript', 'nextjs', 'prisma', 'react', 'css'].map(name =>
      prisma.tag.create({ data: { name } })
    )
  )

  for (const user of users) {
    for (let j = 0; j < 3; j++) {
      const post = await prisma.post.create({
        data: {
          title: `Post ${j + 1} by ${user.name}`,
          content: `This is the content of post ${j + 1} by ${user.name}.`,
          authorId: user.id,
          published: j % 2 === 0,
        },
      })
      await prisma.postTag.create({
        data: { postId: post.id, tagId: tags[j % tags.length].id },
      })
      await prisma.comment.create({
        data: {
          body: `Great post! Comment on post ${post.id}`,
          postId: post.id,
          authorId: users[(user.id % users.length) + 1]?.id || users[0].id,
        },
      })
    }
  }

  console.log('Database seeded successfully')
}

main()
  .catch(console.error)
  .finally(() => prisma.$disconnect())""",
        domain="prisma",
        difficulty="intermediate",
        tags=["seeding", "dev-tools", "test-data"],
    ),
    PromptTemplate(
        instruction="Create a loading skeleton component for a blog post list that matches the layout perfectly",
        context={
            "constraints": [
                "React",
                "Tailwind CSS",
                "Animated pulse",
                "Multiple skeleton items",
            ],
        },
        output_template="""export default function PostListSkeleton({ count = 5 }: { count?: number }) {
  return (
    <div className="space-y-4" role="status" aria-label="Loading posts">
      {[...Array(count)].map((_, i) => (
        <div
          key={i}
          className="border rounded-lg p-6 animate-pulse"
        >
          <div className="h-6 bg-gray-200 rounded w-3/4 mb-4" />
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded w-full" />
            <div className="h-4 bg-gray-200 rounded w-5/6" />
          </div>
          <div className="flex gap-4 mt-4">
            <div className="h-4 bg-gray-200 rounded w-20" />
            <div className="h-4 bg-gray-200 rounded w-16" />
          </div>
        </div>
      ))}
      <span className="sr-only">Loading...</span>
    </div>
  )
}""",
        domain="react",
        difficulty="easy",
        tags=["skeleton", "loading", "tailwind", "accessibility"],
    ),
    PromptTemplate(
        instruction="Debug this Next.js error: 'Error: Functions cannot be passed directly to Client Components' and provide the fix",
        context={
            "problem": "// Server Component\nimport ClientComponent from './ClientComponent'\n\nexport default async function Page() {\n  const data = await fetchData()\n  return <ClientComponent onSave={handleSave} data={data} />\n}",
            "constraints": ["App Router", "Server/Client component boundary"],
        },
        output_template="""Root cause: Server Components cannot pass functions as props to Client Components because functions are not serializable across the server/client boundary.

Fix: Convert the function call into a Server Action and pass the action reference:

```tsx
// app/page.tsx (Server Component)
import ClientComponent from './ClientComponent'
import { saveData } from './actions'

export default async function Page() {
  const data = await fetchData()
  return <ClientComponent saveAction={saveData} data={data} />
}

// app/actions.ts
'use server'
import { prisma } from '@/lib/db'

export async function saveData(formData: FormData) {
  const data = Object.fromEntries(formData)
  await prisma.item.create({ data })
  revalidatePath('/')
}
```

Verify: The component should now render without the error, and the save action is called on the server when the form is submitted.""",
        domain="nextjs",
        difficulty="intermediate",
        tags=["debugging", "server-components", "client-components", "serialization"],
    ),
    PromptTemplate(
        instruction="Create the `lib/db.ts` file that initializes Prisma Client as a singleton for Next.js",
        context={
            "constraints": [
                "Prisma",
                "TypeScript",
                "Singleton pattern",
                "GlobalThis for HMR",
            ],
        },
        output_template="""import { PrismaClient } from '@prisma/client'

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined
}

export const prisma = globalForPrisma.prisma ?? new PrismaClient({
  log: process.env.NODE_ENV === 'development'
    ? ['query', 'error', 'warn']
    : ['error'],
})

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma
}""",
        domain="prisma",
        difficulty="easy",
        tags=["singleton", "setup", "hot-reload", "best-practice"],
    ),
]


def generate_dataset(
    templates: list[PromptTemplate],
    num_variations: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Generate dataset rows from templates with capacity for variations."""
    random.seed(seed)
    rows = []

    for template in templates:
        rows.append(
            {
                "instruction": template.instruction,
                "input": json.dumps(template.context),
                "output": template.output_template,
                "domain": template.domain,
                "difficulty": template.difficulty,
                "tags": template.tags,
            }
        )

    return rows


def write_dataset(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic web dev dataset.")
    parser.add_argument("--output", default="data/domains/nextjs/raw/fullstack.jsonl")
    parser.add_argument(
        "--domain",
        default="all",
        choices=["all", "nextjs", "react", "prisma", "typescript"],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_templates = NEXTJS_TEMPLATES
    if args.domain != "all":
        all_templates = [t for t in all_templates if t.domain == args.domain]

    rows = generate_dataset(all_templates, seed=args.seed)
    write_dataset(rows, Path(args.output))
    print(f"Generated {len(rows)} examples -> {args.output}")

    stats: dict[str, dict] = {}
    for row in rows:
        domain = row.get("domain", "unknown")
        diff = row.get("difficulty", "unknown")
        if domain not in stats:
            stats[domain] = {"count": 0, "difficulties": {}}
        stats[domain]["count"] += 1
        stats[domain]["difficulties"][diff] = (
            stats[domain]["difficulties"].get(diff, 0) + 1
        )

    print("\nDistribution:")
    for domain, s in sorted(stats.items()):
        print(f"  {domain}: {s['count']} rows — {s['difficulties']}")


if __name__ == "__main__":
    main()
