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


BEGINNER_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create a basic HTML5 page with a header, main section, and footer",
        context={"constraints": ["Semantic tags", "Accessible structure"]},
        output_template="""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Starter Page</title>
  </head>
  <body>
    <header>
      <h1>My Starter Page</h1>
    </header>
    <main>
      <section>
        <h2>Welcome</h2>
        <p>This is a beginner HTML layout.</p>
      </section>
    </main>
    <footer>
      <p>Copyright 2026</p>
    </footer>
  </body>
</html>""",
        domain="html",
        difficulty="easy",
        tags=["html", "semantic", "structure"],
    ),
    PromptTemplate(
        instruction="Style a card component with CSS using padding, border radius, and box shadow",
        context={"constraints": ["Plain CSS", "Readable spacing"]},
        output_template=""".card {
  max-width: 320px;
  padding: 16px;
  border-radius: 12px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
}

.card h2 {
  margin: 0 0 8px;
  font-size: 1.2rem;
}

.card p {
  margin: 0;
  color: #4b5563;
}""",
        domain="css",
        difficulty="easy",
        tags=["css", "card", "styling"],
    ),
    PromptTemplate(
        instruction="Create a responsive two-column layout using CSS Grid that collapses to one column on mobile",
        context={"constraints": ["Grid", "Mobile first"]},
        output_template=""".layout {
  display: grid;
  grid-template-columns: 1fr;
  gap: 16px;
}

@media (min-width: 768px) {
  .layout {
    grid-template-columns: 2fr 1fr;
  }
}""",
        domain="css",
        difficulty="easy",
        tags=["css", "grid", "responsive"],
    ),
    PromptTemplate(
        instruction="Build an HTML form for signup with name, email, password, and submit button",
        context={"constraints": ["Use labels", "Basic validation attributes"]},
        output_template="""<form action=\"/signup\" method=\"post\">
  <label for=\"name\">Name</label>
  <input id=\"name\" name=\"name\" type=\"text\" required />

  <label for=\"email\">Email</label>
  <input id=\"email\" name=\"email\" type=\"email\" required />

  <label for=\"password\">Password</label>
  <input id=\"password\" name=\"password\" type=\"password\" minlength=\"8\" required />

  <button type=\"submit\">Create account</button>
</form>""",
        domain="html",
        difficulty="easy",
        tags=["html", "forms", "accessibility"],
    ),
    PromptTemplate(
        instruction="Explain and show safe terminal commands to create a project folder and an index.html file",
        context={"constraints": ["Do not use sudo", "Beginner-safe sequence"]},
        output_template="""# Create project folder
mkdir my-site
cd my-site

# Create starter files
touch index.html styles.css

# Check results
ls -la""",
        domain="terminal",
        difficulty="easy",
        tags=["terminal", "filesystem", "beginner"],
    ),
    PromptTemplate(
        instruction="Show beginner-safe command to copy a file and verify it exists",
        context={"constraints": ["No destructive actions"]},
        output_template="""cp index.html index.backup.html
ls -la index.backup.html""",
        domain="terminal",
        difficulty="easy",
        tags=["terminal", "copy", "verification"],
    ),
    PromptTemplate(
        instruction="Create a simple navigation bar in HTML and style it with CSS flexbox",
        context={"constraints": ["Basic desktop+mobile readiness"]},
        output_template="""<nav class=\"nav\">
  <a href=\"#\">Home</a>
  <a href=\"#\">About</a>
  <a href=\"#\">Contact</a>
</nav>

<style>
  .nav {
    display: flex;
    gap: 12px;
    padding: 12px;
    background: #0f172a;
  }
  .nav a {
    color: #f8fafc;
    text-decoration: none;
  }
</style>""",
        domain="html",
        difficulty="easy",
        tags=["html", "css", "flexbox"],
    ),
]


BACKEND_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create an Express.js REST API with CRUD endpoints for a todo resource using TypeScript",
        context={
            "constraints": [
                "Express 4",
                "TypeScript",
                "In-memory store",
                "Error handling",
            ],
        },
        output_template="""import express, { Request, Response, NextFunction } from 'express'

const app = express()
app.use(express.json())

interface Todo {
  id: number
  title: string
  completed: boolean
}

let todos: Todo[] = []
let nextId = 1

// GET /todos
app.get('/todos', (_req: Request, res: Response) => {
  res.json(todos)
})

// POST /todos
app.post('/todos', (req: Request, res: Response) => {
  const { title } = req.body
  if (!title || typeof title !== 'string') {
    return res.status(400).json({ error: 'Title is required' })
  }
  const todo: Todo = { id: nextId++, title, completed: false }
  todos.push(todo)
  res.status(201).json(todo)
})

// PUT /todos/:id
app.put('/todos/:id', (req: Request, res: Response) => {
  const id = parseInt(req.params.id)
  const todo = todos.find(t => t.id === id)
  if (!todo) return res.status(404).json({ error: 'Not found' })
  if (typeof req.body.completed === 'boolean') todo.completed = req.body.completed
  if (typeof req.body.title === 'string') todo.title = req.body.title
  res.json(todo)
})

// DELETE /todos/:id
app.delete('/todos/:id', (req: Request, res: Response) => {
  const id = parseInt(req.params.id)
  const idx = todos.findIndex(t => t.id === id)
  if (idx === -1) return res.status(404).json({ error: 'Not found' })
  todos.splice(idx, 1)
  res.status(204).send()
})

app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error(err)
  res.status(500).json({ error: 'Internal server error' })
})

app.listen(3000, () => console.log('Server on port 3000'))""",
        domain="express",
        difficulty="easy",
        tags=["express", "typescript", "rest-api", "crud"],
    ),
    PromptTemplate(
        instruction="Create a FastAPI application with async database operations using SQLAlchemy and Pydantic schemas",
        context={
            "constraints": ["FastAPI", "SQLAlchemy 2.0 async", "Pydantic v2"],
        },
        output_template="""from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select
from pydantic import BaseModel, ConfigDict
from typing import Optional
import uvicorn

DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/db"

engine = create_async_engine(DATABASE_URL)
async_session = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class ItemModel(Base):
    __tablename__ = "items"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    price: Mapped[float]
    description: Mapped[Optional[str]]

class ItemCreate(BaseModel):
    name: str
    price: float
    description: Optional[str] = None

class ItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    price: float
    description: Optional[str] = None

app = FastAPI()

async def get_db():
    async with async_session() as session:
        yield session

@app.post("/items", response_model=ItemResponse, status_code=201)
async def create_item(item: ItemCreate, db: AsyncSession = Depends(get_db)):
    db_item = ItemModel(**item.model_dump())
    db.add(db_item)
    await db.commit()
    await db.refresh(db_item)
    return db_item

@app.get("/items", response_model=list[ItemResponse])
async def list_items(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ItemModel))
    return result.scalars().all()

@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(ItemModel).where(ItemModel.id == item_id))
    item = result.scalar_one_or_none()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item""",
        domain="fastapi",
        difficulty="intermediate",
        tags=["fastapi", "sqlalchemy", "async", "pydantic"],
    ),
    PromptTemplate(
        instruction="Create a Flask blueprint for a user authentication API with JWT tokens",
        context={
            "constraints": [
                "Flask 2.x",
                "PyJWT",
                "Blueprint pattern",
                "SQLite via sqlite3",
            ],
        },
        output_template="""from flask import Blueprint, request, jsonify
import sqlite3
import jwt
import hashlib
from datetime import datetime, timedelta

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')
SECRET = 'your-secret-key-change-in-production'

def get_db():
    conn = sqlite3.connect('app.db')
    conn.row_factory = sqlite3.Row
    return conn

@auth_bp.post('/register')
def register():
    data = request.get_json()
    if not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password required'}), 400

    conn = get_db()
    existing = conn.execute('SELECT id FROM users WHERE email = ?', (data['email'],)).fetchone()
    if existing:
        return jsonify({'error': 'Email already registered'}), 409

    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    conn.execute(
        'INSERT INTO users (email, password_hash) VALUES (?, ?)',
        (data['email'], password_hash),
    )
    conn.commit()
    return jsonify({'message': 'User registered'}), 201

@auth_bp.post('/login')
def login():
    data = request.get_json()
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE email = ?', (data.get('email', ''),)).fetchone()
    if not user:
        return jsonify({'error': 'Invalid credentials'}), 401

    password_hash = hashlib.sha256(data['password'].encode()).hexdigest()
    if password_hash != user['password_hash']:
        return jsonify({'error': 'Invalid credentials'}), 401

    token = jwt.encode({
        'user_id': user['id'],
        'email': user['email'],
        'exp': datetime.utcnow() + timedelta(hours=24),
    }, SECRET, algorithm='HS256')

    return jsonify({'token': token})""",
        domain="flask",
        difficulty="intermediate",
        tags=["flask", "blueprint", "jwt", "auth"],
    ),
    PromptTemplate(
        instruction="Create a Go HTTP server with middleware for logging and request ID propagation",
        context={
            "constraints": ["Go 1.21+", "net/http", "No external frameworks"],
        },
        output_template="""package main

import (
    "context"
    "encoding/json"
    "log"
    "net/http"
    "time"

    "github.com/google/uuid"
)

type contextKey string
const requestIDKey contextKey = "request_id"

func withRequestID(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        id := uuid.New().String()
        ctx := context.WithValue(r.Context(), requestIDKey, id)
        w.Header().Set("X-Request-ID", id)
        next.ServeHTTP(w, r.WithContext(ctx))
    })
}

func withLogging(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        start := time.Now()
        next.ServeHTTP(w, r)
        rid, _ := r.Context().Value(requestIDKey).(string)
        log.Printf("[%s] %s %s %v", rid, r.Method, r.URL.Path, time.Since(start))
    })
}

type Todo struct {
    ID        string `json:"id"`
    Title     string `json:"title"`
    Completed bool   `json:"completed"`
}

func handleTodos(w http.ResponseWriter, r *http.Request) {
    todos := []Todo{
        {ID: "1", Title: "Learn Go", Completed: false},
    }
    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(todos)
}

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/todos", handleTodos)

    handler := withRequestID(withLogging(mux))

    srv := &http.Server{
        Addr:         ":8080",
        Handler:      handler,
        ReadTimeout:  10 * time.Second,
        WriteTimeout: 10 * time.Second,
    }

    log.Println("Server starting on :8080")
    log.Fatal(srv.ListenAndServe())
}""",
        domain="go",
        difficulty="intermediate",
        tags=["go", "middleware", "logging", "request-id"],
    ),
    PromptTemplate(
        instruction="Create a Rust Actix-web API endpoint for file upload with size validation and UUID naming",
        context={
            "constraints": ["Actix-web 4", "multipart", "uuid", "tokio async"],
        },
        output_template="""use actix_multipart::Multipart;
use actix_web::{web, App, HttpResponse, HttpServer, post};
use futures_util::StreamExt;
use tokio::fs;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

const MAX_SIZE: usize = 10 * 1024 * 1024; // 10 MB

#[post("/upload")]
async fn upload_file(mut payload: Multipart) -> HttpResponse {
    while let Some(Ok(mut field)) = payload.next().await {
        let content_disposition = field.content_disposition().clone();
        let filename = content_disposition
            .get_filename()
            .map(|f| sanitize_filename::sanitize(f))
            .unwrap_or_else(|| "unnamed".to_string());

        let ext = std::path::Path::new(&filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin");
        let saved_name = format!("{}.{}", Uuid::new_v4(), ext);
        let filepath = format!("./uploads/{}", saved_name);

        let mut file = match fs::File::create(&filepath).await {
            Ok(f) => f,
            Err(e) => {
                return HttpResponse::InternalServerError()
                    .json(serde_json::json!({"error": e.to_string()}))
            }
        };

        let mut size: usize = 0;
        while let Some(Ok(chunk)) = field.next().await {
            size += chunk.len();
            if size > MAX_SIZE {
                let _ = fs::remove_file(&filepath).await;
                return HttpResponse::BadRequest()
                    .json(serde_json::json!({"error": "File too large"}));
            }
            if let Err(e) = file.write_all(&chunk).await {
                return HttpResponse::InternalServerError()
                    .json(serde_json::json!({"error": e.to_string()}));
            }
        }

        return HttpResponse::Ok().json(serde_json::json!({
            "filename": saved_name,
            "original": filename,
            "size": size,
        }));
    }
    HttpResponse::BadRequest().json(serde_json::json!({"error": "No file provided"}))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    fs::create_dir_all("./uploads").await?;
    HttpServer::new(|| App::new().service(upload_file))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}""",
        domain="rust",
        difficulty="intermediate",
        tags=["rust", "actix-web", "upload", "multipart"],
    ),
    PromptTemplate(
        instruction="Create a Flutter widget that fetches and displays a list of posts with pull-to-refresh",
        context={
            "constraints": ["Flutter 3.x", "Dart", "http package", "StatefulWidget"],
        },
        output_template="""import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

class Post {
  final int id;
  final String title;
  final String body;

  Post({required this.id, required this.title, required this.body});

  factory Post.fromJson(Map<String, dynamic> json) {
    return Post(
      id: json['id'],
      title: json['title'],
      body: json['body'],
    );
  }
}

class PostListScreen extends StatefulWidget {
  @override
  _PostListScreenState createState() => _PostListScreenState();
}

class _PostListScreenState extends State<PostListScreen> {
  List<Post> _posts = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _fetchPosts();
  }

  Future<void> _fetchPosts() async {
    setState(() => _loading = true);
    try {
      final response = await http.get(
        Uri.parse('https://jsonplaceholder.typicode.com/posts'),
      );
      if (response.statusCode == 200) {
        final List<dynamic> data = json.decode(response.body);
        setState(() {
          _posts = data.map((json) => Post.fromJson(json)).toList();
          _loading = false;
          _error = null;
        });
      } else {
        throw Exception('Failed to load posts');
      }
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return Center(child: CircularProgressIndicator());
    }
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_error!, style: TextStyle(color: Colors.red)),
            TextButton(onPressed: _fetchPosts, child: Text('Retry')),
          ],
        ),
      );
    }
    return RefreshIndicator(
      onRefresh: _fetchPosts,
      child: ListView.builder(
        itemCount: _posts.length,
        itemBuilder: (context, index) {
          final post = _posts[index];
          return ListTile(
            title: Text(post.title, maxLines: 1, overflow: TextOverflow.ellipsis),
            subtitle: Text(post.body, maxLines: 2, overflow: TextOverflow.ellipsis),
          );
        },
      ),
    );
  }
}""",
        domain="flutter",
        difficulty="easy",
        tags=["flutter", "dart", "http", "listview"],
    ),
    PromptTemplate(
        instruction="Create a Dockerfile for a Node.js Express application with multi-stage build for production",
        context={
            "constraints": [
                "Docker",
                "Node 20-alpine",
                "Multi-stage",
                "Security best practices",
            ],
        },
        output_template="""FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production && npm cache clean --force

FROM node:20-alpine AS runner
RUN addgroup -g 1001 appgroup && adduser -u 1001 -G appgroup -s /bin/sh -D appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appgroup /app/node_modules ./node_modules
COPY --chown=appuser:appgroup . .

USER appuser
EXPOSE 3000

ENV NODE_ENV=production
ENV PORT=3000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "dist/server.js"]""",
        domain="docker",
        difficulty="easy",
        tags=["docker", "multi-stage", "nodejs", "security"],
    ),
    PromptTemplate(
        instruction="Create a docker-compose.yml with a Node.js app, PostgreSQL database, and Redis cache",
        context={
            "constraints": [
                "Docker Compose v3.8",
                "PostgreSQL 16",
                "Redis 7-alpine",
                "Health checks",
            ],
        },
        output_template="""version: '3.8'

services:
  app:
    build: .
    container_name: app-server
    ports:
      - '3000:3000'
    environment:
      DATABASE_URL: postgresql://user:password@db:5432/appdb
      REDIS_URL: redis://cache:6379
      NODE_ENV: development
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_healthy
    volumes:
      - .:/app
      - /app/node_modules
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    container_name: app-db
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: appdb
    ports:
      - '5432:5432'
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ['CMD-SHELL', 'pg_isready -U user -d appdb']
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  cache:
    image: redis:7-alpine
    container_name: app-cache
    ports:
      - '6379:6379'
    healthcheck:
      test: ['CMD', 'redis-cli', 'ping']
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  pgdata:""",
        domain="docker",
        difficulty="intermediate",
        tags=["docker", "compose", "postgresql", "redis"],
    ),
    PromptTemplate(
        instruction="Create a GitHub Actions CI workflow for a Node.js project that runs lint, type-check, and tests",
        context={
            "constraints": ["GitHub Actions", "Node 20", "npm", "Cache strategy"],
        },
        output_template="""name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [18, 20, 22]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck

      - name: Test with coverage
        run: npm run test:coverage

      - name: Build
        run: npm run build

      - name: Upload coverage
        if: matrix.node-version == '20'
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage/lcov.info

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - name: Audit dependencies
        run: npm audit --audit-level=high""",
        domain="cicd",
        difficulty="easy",
        tags=["github-actions", "ci", "nodejs", "testing"],
    ),
    PromptTemplate(
        instruction="Create a FastAPI middleware for rate limiting using Redis with sliding window algorithm",
        context={
            "constraints": [
                "FastAPI",
                "Redis async",
                "Sliding window",
                "Per-IP tracking",
            ],
        },
        output_template="""import asyncio
from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, redis_url: str = "redis://localhost:6379",
                 max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        key = f"ratelimit:{client_ip}"
        now = time.time()
        window_start = now - self.window_seconds

        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zadd(key, {str(now): now})
            pipe.expire(key, self.window_seconds + 1)
            _, count, _, _ = await pipe.execute()

        if count > self.max_requests:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later.",
                headers={"Retry-After": str(self.window_seconds)},
            )

        return await call_next(request)

app = FastAPI()
app.add_middleware(
    RateLimitMiddleware,
    max_requests=100,
    window_seconds=60,
)""",
        domain="fastapi",
        difficulty="advanced",
        tags=["fastapi", "rate-limiting", "redis", "middleware"],
    ),
    PromptTemplate(
        instruction="Create a Go CLI tool that reads a CSV file and outputs JSON statistics",
        context={
            "constraints": [
                "Go",
                "encoding/csv",
                "flag for CLI args",
                "Error handling",
            ],
        },
        output_template="""package main

import (
    "encoding/csv"
    "encoding/json"
    "flag"
    "fmt"
    "os"
    "strconv"
)

func main() {
    input := flag.String("input", "data.csv", "Path to input CSV file")
    column := flag.String("column", "", "Column name to analyze")
    flag.Parse()

    if *column == "" {
        fmt.Fprintln(os.Stderr, "Error: --column is required")
        os.Exit(1)
    }

    file, err := os.Open(*input)
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error opening file: %v\n", err)
        os.Exit(1)
    }
    defer file.Close()

    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error reading CSV: %v\n", err)
        os.Exit(1)
    }

    if len(records) < 2 {
        fmt.Fprintln(os.Stderr, "CSV must have a header and at least one row")
        os.Exit(1)
    }

    headers := records[0]
    colIdx := -1
    for i, h := range headers {
        if h == *column {
            colIdx = i
            break
        }
    }
    if colIdx == -1 {
        fmt.Fprintf(os.Stderr, "Column '%s' not found\n", *column)
        os.Exit(1)
    }

    var values []float64
    for _, row := range records[1:] {
        if val, err := strconv.ParseFloat(row[colIdx], 64); err == nil {
            values = append(values, val)
        }
    }

    if len(values) == 0 {
        fmt.Fprintln(os.Stderr, "No valid numeric values found")
        os.Exit(1)
    }

    sum, min, max := values[0], values[0], values[0]
    for _, v := range values[1:] {
        sum += v
        if v < min { min = v }
        if v > max { max = v }
    }
    avg := sum / float64(len(values))

    result := map[string]interface{}{
        "column": *column,
        "count":  len(values),
        "sum":    sum,
        "average": avg,
        "min":    min,
        "max":    max,
    }

    json.NewEncoder(os.Stdout).Encode(result)
}""",
        domain="go",
        difficulty="easy",
        tags=["go", "cli", "csv", "json"],
    ),
    PromptTemplate(
        instruction="Create a Rust CLI that recursively finds duplicate files by SHA-256 hash",
        context={
            "constraints": [
                "Rust",
                "clap for args",
                "sha2",
                "walkdir",
                "Parallel with rayon",
            ],
        },
        output_template="""use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use clap::Parser;
use sha2::{Sha256, Digest};
use walkdir::WalkDir;
use rayon::prelude::*;

#[derive(Parser)]
struct Args {
    #[arg(default_value = ".")]
    directory: String,

    #[arg(short, long, default_value = "1048576")]
    min_size: u64,
}

#[derive(Debug)]
struct FileEntry {
    path: PathBuf,
    size: u64,
}

fn main() {
    let args = Args::parse();

    let files: Vec<FileEntry> = WalkDir::new(&args.directory)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let meta = e.metadata().ok()?;
            if meta.len() >= args.min_size {
                Some(FileEntry { path: e.path().to_owned(), size: meta.len() })
            } else {
                None
            }
        })
        .collect();

    let mut size_map: HashMap<u64, Vec<FileEntry>> = HashMap::new();
    for entry in files {
        size_map.entry(entry.size).or_default().push(entry);
    }

    let candidates: Vec<Vec<FileEntry>> = size_map
        .into_values()
        .filter(|v| v.len() > 1)
        .collect();

    let hashed: Vec<(String, PathBuf)> = candidates
        .par_iter()
        .flat_map(|group| {
            group.iter().filter_map(|entry| {
                let content = fs::read(&entry.path).ok()?;
                let mut hasher = Sha256::new();
                hasher.update(&content);
                let hash = format!("{:x}", hasher.finalize());
                Some((hash, entry.path.clone()))
            }).collect::<Vec<_>>()
        })
        .collect();

    let mut hash_map: HashMap<String, Vec<PathBuf>> = HashMap::new();
    for (hash, path) in hashed {
        hash_map.entry(hash).or_default().push(path);
    }

    println!("{}", serde_json::to_string_pretty(&hash_map).unwrap());
}""",
        domain="rust",
        difficulty="intermediate",
        tags=["rust", "cli", "hashing", "parallel"],
    ),
    PromptTemplate(
        instruction="Create a Kubernetes Deployment and Service manifest for a Node.js microservice",
        context={
            "constraints": [
                "Kubernetes",
                "Deployment with 3 replicas",
                "Readiness/Liveness probes",
                "Resource limits",
            ],
        },
        output_template="""apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels:
    app: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
        - name: api
          image: registry.example.com/api-server:latest
          ports:
            - containerPort: 3000
          env:
            - name: NODE_ENV
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secrets
                  key: url
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: api-server
spec:
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 3000
      protocol: TCP
  type: ClusterIP""",
        domain="kubernetes",
        difficulty="intermediate",
        tags=["kubernetes", "deployment", "service", "probes"],
    ),
    PromptTemplate(
        instruction="Create a Kubernetes ConfigMap and Secret for a web application",
        context={
            "constraints": [
                "Kubernetes",
                "ConfigMap from literal values",
                "Secret with base64 values",
            ],
        },
        output_template="""apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_NAME: "my-web-app"
  APP_ENV: "production"
  LOG_LEVEL: "info"
  API_TIMEOUT_MS: "5000"
  CORS_ORIGINS: "https://app.example.com,https://admin.example.com"
---
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:password@db:5432/appdb"
  REDIS_URL: "redis://:password@cache:6379"
  JWT_SECRET: "your-production-jwt-secret-min-32-chars"
  API_KEY: "sk-prod-api-key-1234567890abcdef"
---
apiVersion: v1
kind: Pod
metadata:
  name: app-pod
spec:
  containers:
    - name: app
      image: my-app:latest
      envFrom:
        - configMapRef:
            name: app-config
      env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: DATABASE_URL
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: JWT_SECRET""",
        domain="kubernetes",
        difficulty="easy",
        tags=["kubernetes", "configmap", "secret", "env"],
    ),
    PromptTemplate(
        instruction="Create a Python script that uses asyncio to concurrently fetch multiple URLs and aggregate results",
        context={
            "constraints": ["Python 3.10+", "asyncio", "aiohttp", "concurrent limits"],
        },
        output_template="""import asyncio
import aiohttp
from typing import Any

async def fetch_url(session: aiohttp.ClientSession, url: str, sem: asyncio.Semaphore) -> dict[str, Any]:
    async with sem:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                text = await resp.text()
                return {"url": url, "status": resp.status, "length": len(text), "error": None}
        except Exception as e:
            return {"url": url, "status": 0, "length": 0, "error": str(e)}

async def fetch_all(urls: list[str], concurrency: int = 5) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url, sem) for url in urls]
        return await asyncio.gather(*tasks)

async def main():
    urls = [
        "https://httpbin.org/get",
        "https://httpbin.org/delay/1",
        "https://httpbin.org/status/404",
        "https://invalid.example.com",
    ]
    results = await fetch_all(urls, concurrency=3)
    for r in results:
        status_icon = "OK" if r["status"] == 200 else f"ERR({r['status']})"
        print(f"[{status_icon}] {r['url']}: {r['length']} bytes")
        if r["error"]:
            print(f"  Error: {r['error']}")

if __name__ == "__main__":
    asyncio.run(main())""",
        domain="python",
        difficulty="intermediate",
        tags=["python", "asyncio", "aiohttp", "concurrent"],
    ),
    PromptTemplate(
        instruction="Create a Python function that implements a thread-safe LRU cache with TTL expiration",
        context={
            "constraints": ["Python", "threading", "OrderedDict", "TTL per entry"],
        },
        output_template="""import threading
import time
from collections import OrderedDict
from typing import Any, TypeVar

K = TypeVar('K')
V = TypeVar('V')

class TTLLRUCache:
    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.monotonic() - timestamp > self.ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.monotonic())
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def delete(self, key: K) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear_expired(self) -> int:
        with self._lock:
            now = time.monotonic()
            expired = [k for k, (_, ts) in self._cache.items() if now - ts > self.ttl]
            for k in expired:
                del self._cache[k]
            return len(expired)

    def __len__(self) -> int:
        return len(self._cache)""",
        domain="python",
        difficulty="advanced",
        tags=["python", "cache", "lru", "thread-safe"],
    ),
    PromptTemplate(
        instruction="Create a Python decorator that retries a function with exponential backoff on specified exceptions",
        context={
            "constraints": ["Python", "decorator", "exponential backoff", "jitter"],
        },
        output_template="""import time
import random
import functools
from typing import Type, Callable, Any

def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    jitter: bool = True,
):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts:
                        raise
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    print(f"[Attempt {attempt}/{max_attempts}] Retrying in {delay:.1f}s: {e}")
                    time.sleep(delay)
            raise last_exception  # type: ignore
        return wrapper
    return decorator

@retry(max_attempts=5, base_delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_external_api(url: str) -> dict:
    import requests  # simulate
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    return response.json()""",
        domain="python",
        difficulty="intermediate",
        tags=["python", "decorator", "retry", "backoff"],
    ),
    PromptTemplate(
        instruction="Create a Go rate limiter using the token bucket algorithm with middleware for HTTP handlers",
        context={
            "constraints": [
                "Go",
                "Token bucket",
                "net/http middleware",
                "sync package",
            ],
        },
        output_template="""package main

import (
    "net/http"
    "sync"
    "time"
)

type TokenBucket struct {
    rate       float64
    capacity   float64
    tokens     float64
    lastRefill time.Time
    mu         sync.Mutex
}

func NewTokenBucket(rate float64, capacity float64) *TokenBucket {
    return &TokenBucket{
        rate:       rate,
        capacity:   capacity,
        tokens:     capacity,
        lastRefill: time.Now(),
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.mu.Lock()
    defer tb.mu.Unlock()

    now := time.Now()
    elapsed := now.Sub(tb.lastRefill).Seconds()
    tb.tokens = min(tb.capacity, tb.tokens+elapsed*tb.rate)
    tb.lastRefill = now

    if tb.tokens >= 1 {
        tb.tokens--
        return true
    }
    return false
}

func RateLimitMiddleware(bucket *TokenBucket) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            if !bucket.Allow() {
                w.Header().Set("Retry-After", "1")
                http.Error(w, `{"error":"rate limit exceeded"}`, http.StatusTooManyRequests)
                return
            }
            next.ServeHTTP(w, r)
        })
    }
}

func main() {
    limiter := NewTokenBucket(100, 100) // 100 req/s with burst of 100
    mux := http.NewServeMux()
    mux.HandleFunc("/api/data", func(w http.ResponseWriter, r *http.Request) {
        w.Write([]byte(`{"data":"ok"}`))
    })
    http.ListenAndServe(":8080", RateLimitMiddleware(limiter)(mux))
}""",
        domain="go",
        difficulty="advanced",
        tags=["go", "rate-limiting", "token-bucket", "middleware"],
    ),
    PromptTemplate(
        instruction="Create a TypeScript generic Result type for railway-oriented error handling with map and flatMap",
        context={
            "constraints": [
                "TypeScript strict",
                "Generic",
                "Discriminated union",
                "Method chaining",
            ],
        },
        output_template="""type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E }

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value }
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error }
}

function map<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E> {
  if (result.ok) return ok(fn(result.value))
  return result
}

function flatMap<T, U, E>(result: Result<T, E>, fn: (value: T) => Result<U, E>): Result<U, E> {
  if (result.ok) return fn(result.value)
  return result
}

function match<T, U, E>(
  result: Result<T, E>,
  handlers: { ok: (value: T) => U; err: (error: E) => U },
): U {
  return result.ok ? handlers.ok(result.value) : handlers.err(result.error)
}

// Example usage
async function fetchUser(id: number): Promise<Result<{ name: string }>> {
  try {
    const res = await fetch(`/api/users/${id}`)
    if (!res.ok) return err(new Error(`HTTP ${res.status}`))
    return ok(await res.json())
  } catch (e) {
    return err(e instanceof Error ? e : new Error(String(e)))
  }
}

type AsyncResult<T, E = Error> = Promise<Result<T, E>>

async function main() {
  const result = await fetchUser(1)
  match(result, {
    ok: (user) => console.log('User:', user.name),
    err: (error) => console.error('Failed:', error.message),
  })
}""",
        domain="typescript",
        difficulty="intermediate",
        tags=["typescript", "result-type", "error-handling", "generics"],
    ),
    PromptTemplate(
        instruction="Create a Go worker pool pattern that processes jobs concurrently with a bounded channel",
        context={
            "constraints": [
                "Go",
                "goroutines",
                "channels",
                "sync.WaitGroup",
                "context cancellation",
            ],
        },
        output_template="""package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type Job struct {
    ID   int
    Data string
}

type Result struct {
    JobID  int
    Output string
    Error  error
}

func WorkerPool(ctx context.Context, jobs <-chan Job, results chan<- Result, workerCount int) {
    var wg sync.WaitGroup

    for i := range workerCount {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for {
                select {
                case <-ctx.Done():
                    return
                case job, ok := <-jobs:
                    if !ok {
                        return
                    }
                    output, err := processJob(job)
                    select {
                    case results <- Result{JobID: job.ID, Output: output, Error: err}:
                    case <-ctx.Done():
                        return
                    }
                }
            }
        }(i)
    }

    wg.Wait()
    close(results)
}

func processJob(job Job) (string, error) {
    time.Sleep(100 * time.Millisecond)
    return fmt.Sprintf("processed: %s", job.Data), nil
}

func main() {
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    jobs := make(chan Job, 10)
    results := make(chan Result, 10)

    go WorkerPool(ctx, jobs, results, 4)

    for i := range 20 {
        select {
        case <-ctx.Done():
            break
        case jobs <- Job{ID: i, Data: fmt.Sprintf("job-%d", i)}:
        }
    }
    close(jobs)

    for result := range results {
        if result.Error != nil {
            fmt.Printf("Job %d failed: %v\n", result.JobID, result.Error)
        } else {
            fmt.Printf("Job %d: %s\n", result.JobID, result.Output)
        }
    }
}""",
        domain="go",
        difficulty="advanced",
        tags=["go", "worker-pool", "goroutines", "context"],
    ),
]

DEVOPS_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create a GitHub Actions workflow that builds and publishes a Docker image to GitHub Container Registry",
        context={
            "constraints": [
                "GitHub Actions",
                "Docker Buildx",
                "GHCR",
                "Multi-platform build",
            ],
        },
        output_template="""name: Build and Publish Docker Image

on:
  push:
    tags: ['v*']
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=,format=short

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          platforms: linux/amd64,linux/arm64
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max""",
        domain="cicd",
        difficulty="intermediate",
        tags=["github-actions", "docker", "ghcr", "multi-platform"],
    ),
    PromptTemplate(
        instruction="Create a Makefile for a Go project with build, test, lint, and docker targets",
        context={
            "constraints": ["Make", "Go 1.22+", "Docker", "golangci-lint"],
        },
        output_template="""APP_NAME := myapp
VERSION  := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
LDFLAGS  := -ldflags="-X main.version=$(VERSION)"
BIN_DIR  := ./bin
OUT      := $(BIN_DIR)/$(APP_NAME)

.PHONY: build run test lint clean docker-build docker-run help

build: ## Build the binary
    @mkdir -p $(BIN_DIR)
    go build $(LDFLAGS) -o $(OUT) ./cmd/$(APP_NAME)

run: build ## Build and run locally
    $(OUT)

test: ## Run tests with coverage
    go test -race -coverprofile=coverage.out -covermode=atomic ./...
    go tool cover -func=coverage.out

test-integration: ## Run integration tests
    go test -tags=integration -count=1 ./...

lint: ## Run static analysis
    golangci-lint run ./...

fmt: ## Format code
    go fmt ./...

clean: ## Remove build artifacts
    rm -rf $(BIN_DIR) coverage.out

docker-build: ## Build Docker image
    docker build -t $(APP_NAME):$(VERSION) .
    docker tag $(APP_NAME):$(VERSION) $(APP_NAME):latest

docker-run: docker-build ## Build and run Docker container
    docker run --rm -p 8080:8080 $(APP_NAME):latest

help: ## Show this help
    @grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
        awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'""",
        domain="devops",
        difficulty="easy",
        tags=["makefile", "go", "docker", "build"],
    ),
    PromptTemplate(
        instruction="Create a Terraform configuration for an AWS ECS Fargate service with ALB and autoscaling",
        context={
            "constraints": ["Terraform", "AWS", "ECS Fargate", "ALB", "Autoscaling"],
        },
        output_template="""terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

resource "aws_ecs_cluster" "main" {
  name = "app-cluster"
}

resource "aws_ecs_task_definition" "app" {
  family                   = "app-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name  = "app"
    image = "${aws_ecr_repository.app.repository_url}:latest"
    portMappings = [{ containerPort = 3000, protocol = "tcp" }]
    environment = [
      { name = "NODE_ENV", value = "production" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.app.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "app"
      }
    }
  }])
}

resource "aws_ecs_service" "app" {
  name            = "app-service"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "app"
    container_port   = 3000
  }

  depends_on = [aws_lb_listener.app]
}

resource "aws_appautoscaling_target" "app" {
  max_capacity       = 10
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.app.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cpu" {
  name               = "app-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.app.resource_id
  scalable_dimension = aws_appautoscaling_target.app.scalable_dimension
  service_namespace  = aws_appautoscaling_target.app.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    scale_in_cooldown  = 60
    scale_out_cooldown = 60
  }
}""",
        domain="devops",
        difficulty="advanced",
        tags=["terraform", "aws", "ecs", "fargate", "autoscaling"],
    ),
]

MOBILE_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create a Flutter state management solution using ChangeNotifier and Provider for a shopping cart",
        context={
            "constraints": ["Flutter", "Provider", "ChangeNotifier", "Type-safe"],
        },
        output_template="""import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

class CartItem {
  final String id;
  final String name;
  final double price;
  int quantity;
  CartItem({required this.id, required this.name, required this.price, this.quantity = 1});
}

class CartModel extends ChangeNotifier {
  final List<CartItem> _items = [];

  List<CartItem> get items => List.unmodifiable(_items);
  int get itemCount => _items.fold(0, (sum, item) => sum + item.quantity);
  double get totalPrice => _items.fold(0, (sum, item) => sum + item.price * item.quantity);

  void addItem(CartItem item) {
    final existing = _items.where((i) => i.id == item.id);
    if (existing.isNotEmpty) {
      existing.first.quantity += item.quantity;
    } else {
      _items.add(item);
    }
    notifyListeners();
  }

  void removeItem(String id) {
    _items.removeWhere((item) => item.id == id);
    notifyListeners();
  }

  void updateQuantity(String id, int quantity) {
    if (quantity <= 0) {
      removeItem(id);
      return;
    }
    _items.firstWhere((item) => item.id == id).quantity = quantity;
    notifyListeners();
  }

  void clear() {
    _items.clear();
    notifyListeners();
  }
}

void main() {
  runApp(
    ChangeNotifierProvider(
      create: (_) => CartModel(),
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text('Shop'),
          actions: [
            Consumer<CartModel>(
              builder: (_, cart, __) => Stack(
                children: [
                  Icon(Icons.shopping_cart),
                  if (cart.itemCount > 0)
                    Positioned(
                      right: 0,
                      child: Container(
                        padding: EdgeInsets.all(2),
                        decoration: BoxDecoration(
                          color: Colors.red,
                          shape: BoxShape.circle,
                        ),
                        child: Text('${cart.itemCount}',
                          style: TextStyle(fontSize: 11, color: Colors.white)),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
        body: Center(child: Text('Shop content')),
      ),
    );
  }
}""",
        domain="flutter",
        difficulty="intermediate",
        tags=["flutter", "provider", "state-management", "cart"],
    ),
]

LANG_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Create a Java Spring Boot REST controller with CRUD operations for a Product entity using JPA repository",
        context={
            "constraints": ["Spring Boot 3", "Java 17", "JPA", "Lombok"],
        },
        output_template="""@RestController
@RequestMapping("/api/products")
@RequiredArgsConstructor
public class ProductController {
    private final ProductRepository repository;

    @GetMapping
    public List<Product> findAll() {
        return repository.findAll();
    }

    @GetMapping("/{id}")
    public ResponseEntity<Product> findById(@PathVariable Long id) {
        return repository.findById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    @ResponseStatus(HttpStatus.CREATED)
    public Product create(@Valid @RequestBody Product product) {
        return repository.save(product);
    }

    @PutMapping("/{id}")
    public ResponseEntity<Product> update(@PathVariable Long id, @Valid @RequestBody Product updated) {
        return repository.findById(id).map(existing -> {
            existing.setName(updated.getName());
            existing.setPrice(updated.getPrice());
            return ResponseEntity.ok(repository.save(existing));
        }).orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        if (repository.existsById(id)) {
            repository.deleteById(id);
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
class Product {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    private String name;
    private BigDecimal price;
}

interface ProductRepository extends JpaRepository<Product, Long> {}""",
        domain="java",
        difficulty="easy",
        tags=["java", "spring-boot", "rest", "jpa"],
    ),
    PromptTemplate(
        instruction="Create a C# ASP.NET Core minimal API with Entity Framework Core for a Todo app with SQLite",
        context={
            "constraints": ["C# 12", ".NET 8", "EF Core", "SQLite"],
        },
        output_template="""using Microsoft.EntityFrameworkCore;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddDbContext<TodoDb>(opt => opt.UseSqlite("Data Source=todos.db"));
var app = builder.Build();

app.MapGet("/todos", async (TodoDb db) =>
    await db.Todos.ToListAsync());

app.MapGet("/todos/{id}", async (int id, TodoDb db) =>
    await db.Todos.FindAsync(id) is Todo todo ? Results.Ok(todo) : Results.NotFound());

app.MapPost("/todos", async (Todo todo, TodoDb db) => {
    db.Todos.Add(todo);
    await db.SaveChangesAsync();
    return Results.Created($"/todos/{todo.Id}", todo);
});

app.MapPut("/todos/{id}", async (int id, Todo input, TodoDb db) => {
    var todo = await db.Todos.FindAsync(id);
    if (todo is null) return Results.NotFound();
    todo.Title = input.Title;
    todo.IsCompleted = input.IsCompleted;
    await db.SaveChangesAsync();
    return Results.Ok(todo);
});

app.MapDelete("/todos/{id}", async (int id, TodoDb db) => {
    var todo = await db.Todos.FindAsync(id);
    if (todo is null) return Results.NotFound();
    db.Todos.Remove(todo);
    await db.SaveChangesAsync();
    return Results.NoContent();
});

app.Run();

class Todo {
    public int Id { get; set; }
    public string Title { get; set; } = "";
    public bool IsCompleted { get; set; }
}

class TodoDb : DbContext {
    public TodoDb(DbContextOptions<TodoDb> options) : base(options) {}
    public DbSet<Todo> Todos => Set<Todo>();
}""",
        domain="csharp",
        difficulty="easy",
        tags=["csharp", "dotnet", "ef-core", "minimal-api"],
    ),
    PromptTemplate(
        instruction="Create a Kotlin Ktor server with an in-memory user repository and JSON serialization",
        context={
            "constraints": ["Kotlin 2.0", "Ktor 3", "kotlinx.serialization"],
        },
        output_template="""import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import kotlinx.serialization.Serializable
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicLong

@Serializable
data class User(val id: Long, val name: String, val email: String)

@Serializable
data class CreateUserRequest(val name: String, val email: String)

val users = ConcurrentHashMap<Long, User>()
val nextId = AtomicLong(1)

fun main() {
    embeddedServer(Netty, port = 8080) {
        install(ContentNegotiation) { json() }
        routing {
            route("/users") {
                get {
                    call.respond(users.values.toList())
                }
                get("/{id}") {
                    val id = call.parameters["id"]?.toLongOrNull()
                    val user = id?.let { users[it] }
                    if (user != null) call.respond(user)
                    else call.respondText("Not found", status = HttpStatusCode.NotFound)
                }
                post {
                    val req = call.receive<CreateUserRequest>()
                    val user = User(nextId.getAndIncrement(), req.name, req.email)
                    users[user.id] = user
                    call.respond(HttpStatusCode.Created, user)
                }
                delete("/{id}") {
                    val id = call.parameters["id"]?.toLongOrNull()
                    if (id != null && users.remove(id) != null)
                        call.respond(HttpStatusCode.NoContent)
                    else call.respondText("Not found", status = HttpStatusCode.NotFound)
                }
            }
        }
    }.start(wait = true)
}""",
        domain="kotlin",
        difficulty="easy",
        tags=["kotlin", "ktor", "serialization", "rest"],
    ),
    PromptTemplate(
        instruction="Create a Ruby on Rails 7 API-only controller for a Blog Post resource with strong params and JSON responses",
        context={
            "constraints": ["Rails 7 API mode", "Ruby 3.2", "Jbuilder or plain JSON"],
        },
        output_template="""class Api::V1::PostsController < ApplicationController
  before_action :set_post, only: [:show, :update, :destroy]

  def index
    @posts = Post.order(created_at: :desc).page(params[:page]).per(20)
    render json: {
      posts: @posts.as_json(include: { author: { only: [:id, :name] } }),
      meta: { total: Post.count, page: params[:page]&.to_i || 1 }
    }
  end

  def show
    render json: @post.as_json(include: { author: { only: [:id, :name] }, comments: {} })
  end

  def create
    @post = Post.new(post_params)
    @post.author = current_user

    if @post.save
      render json: @post, status: :created, location: api_v1_post_url(@post)
    else
      render json: { errors: @post.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def update
    if @post.update(post_params)
      render json: @post
    else
      render json: { errors: @post.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def destroy
    @post.destroy!
    head :no_content
  end

  private

  def set_post
    @post = Post.find(params[:id])
  end

  def post_params
    params.require(:post).permit(:title, :content, :published)
  end
end""",
        domain="ruby",
        difficulty="intermediate",
        tags=["ruby", "rails", "api", "json"],
    ),
    PromptTemplate(
        instruction="Create a PHP Laravel controller for a product review system with rate limiting and validation",
        context={
            "constraints": ["Laravel 11", "PHP 8.2", "Eloquent", "Form Request"],
        },
        output_template="""<?php

namespace App\\Http\\Controllers\\Api;

use App\\Http\\Requests\\StoreReviewRequest;
use App\\Models\\Product;
use App\\Models\\Review;
use Illuminate\\Http\\JsonResponse;
use Illuminate\\Support\\Facades\\RateLimiter;

class ReviewController extends Controller
{
    public function index(Product $product): JsonResponse
    {
        $reviews = $product->reviews()
            ->with('user:id,name')
            ->latest()
            ->paginate(15);

        return response()->json($reviews);
    }

    public function store(StoreReviewRequest $request, Product $product): JsonResponse
    {
        $key = 'review:' . $request->user()->id;

        if (RateLimiter::tooManyAttempts($key, 5)) {
            return response()->json(['message' => 'Too many reviews. Try again later.'], 429);
        }

        RateLimiter::hit($key, 3600);

        $review = $product->reviews()->create([
            'user_id' => $request->user()->id,
            'rating' => $request->rating,
            'title' => $request->title,
            'body' => $request->body,
        ]);

        return response()->json($review->load('user:id,name'), 201);
    }

    public function destroy(Product $product, Review $review): JsonResponse
    {
        if ($review->user_id !== request()->user()->id) {
            return response()->json(['message' => 'Unauthorized'], 403);
        }

        $review->delete();
        return response()->json(null, 204);
    }
}""",
        domain="php",
        difficulty="intermediate",
        tags=["php", "laravel", "eloquent", "rate-limiting"],
    ),
    PromptTemplate(
        instruction="Create a Swift Vapor server-side API endpoint for a workout tracker with structured concurrency",
        context={
            "constraints": ["Swift 6", "Vapor 4", "Fluent", "PostgreSQL"],
        },
        output_template="""import Vapor
import Fluent

struct WorkoutController: RouteCollection {
    func boot(routes: RoutesBuilder) throws {
        let workouts = routes.grouped("api", "workouts")
        workouts.get(use: index)
        workouts.post(use: create)
        workouts.group(":workoutID") { workout in
            workout.get(use: show)
            workout.delete(use: delete)
        }
    }

    @Sendable
    func index(req: Request) async throws -> [Workout] {
        let userId = try req.auth.require(User.self).requireID()
        return try await Workout.query(on: req.db)
            .filter(\\.$user.$id == userId)
            .sort(\\.$date, .descending)
            .all()
    }

    @Sendable
    func create(req: Request) async throws -> Workout {
        let userId = try req.auth.require(User.self).requireID()
        let input = try req.content.decode(CreateWorkout.self)

        let workout = Workout(
            userId: userId,
            name: input.name,
            durationMinutes: input.durationMinutes,
            date: input.date ?? Date()
        )

        try await req.db.transaction { db in
            try await workout.save(on: db)
            for exercise in input.exercises {
                let ex = Exercise(workoutId: workout.requireID(), name: exercise.name, sets: exercise.sets, reps: exercise.reps)
                try await ex.save(on: db)
            }
        }

        return workout
    }

    @Sendable
    func show(req: Request) async throws -> Workout {
        guard let workout = try await Workout.find(req.parameters.get("workoutID"), on: req.db) else {
            throw Abort(.notFound)
        }
        try await workout.$exercises.load(on: req.db)
        return workout
    }

    @Sendable
    func delete(req: Request) async throws -> HTTPStatus {
        guard let workout = try await Workout.find(req.parameters.get("workoutID"), on: req.db) else {
            throw Abort(.notFound)
        }
        try await workout.delete(on: req.db)
        return .noContent
    }
}

struct CreateWorkout: Content {
    let name: String
    let durationMinutes: Int
    let date: Date?
    let exercises: [CreateExercise]
}
struct CreateExercise: Content {
    let name: String
    let sets: Int
    let reps: Int
}""",
        domain="swift",
        difficulty="intermediate",
        tags=["swift", "vapor", "fluent", "async"],
    ),
    PromptTemplate(
        instruction="Create a comprehensive Ansible playbook for deploying a Python web app with Nginx reverse proxy and systemd service",
        context={
            "constraints": [
                "Ansible 2.15+",
                "Ubuntu 22.04",
                "Idempotent",
                "Uses handlers",
            ],
        },
        output_template="""---
- name: Deploy Python web application
  hosts: web
  become: yes
  vars:
    app_name: myapp
    app_dir: /opt/{{ app_name }}
    app_user: appuser
    python_version: "3.11"
    repo_url: https://github.com/example/myapp.git

  tasks:
    - name: Install system packages
      apt:
        name:
          - python{{ python_version }}
          - python{{ python_version }}-venv
          - python3-pip
          - git
          - nginx
        state: present
        update_cache: yes

    - name: Create application user
      user:
        name: "{{ app_user }}"
        system: yes
        create_home: no
        shell: /usr/sbin/nologin

    - name: Clone application repository
      git:
        repo: "{{ repo_url }}"
        dest: "{{ app_dir }}"
        version: main
        force: yes
      notify: restart app

    - name: Create Python virtual environment
      pip:
        requirements: "{{ app_dir }}/requirements.txt"
        virtualenv: "{{ app_dir }}/venv"
        virtualenv_command: python{{ python_version }} -m venv
      notify: restart app

    - name: Create systemd service unit
      copy:
        dest: /etc/systemd/system/{{ app_name }}.service
        content: |
          [Unit]
          Description={{ app_name }} service
          After=network.target

          [Service]
          Type=simple
          User={{ app_user }}
          WorkingDirectory={{ app_dir }}
          ExecStart={{ app_dir }}/venv/bin/gunicorn app:app -w 4 -b 127.0.0.1:8000
          Restart=always
          RestartSec=5

          [Install]
          WantedBy=multi-user.target
      notify: restart app

    - name: Enable and start service
      systemd:
        name: "{{ app_name }}"
        enabled: yes
        state: started
        daemon_reload: yes

    - name: Configure Nginx reverse proxy
      template:
        src: nginx.conf.j2
        dest: /etc/nginx/sites-available/{{ app_name }}
      notify: reload nginx

    - name: Enable Nginx site
      file:
        src: /etc/nginx/sites-available/{{ app_name }}
        dest: /etc/nginx/sites-enabled/{{ app_name }}
        state: link
      notify: reload nginx

  handlers:
    - name: restart app
      systemd:
        name: "{{ app_name }}"
        state: restarted
        daemon_reload: yes

    - name: reload nginx
      systemd:
        name: nginx
        state: reloaded""",
        domain="devops",
        difficulty="intermediate",
        tags=["ansible", "deployment", "nginx", "systemd"],
    ),
    PromptTemplate(
        instruction="Create a Vue 3 Composition API component with Pinia store for a shopping cart with quantity controls and price calculation",
        context={
            "constraints": ["Vue 3", "Composition API", "Pinia", "TypeScript"],
        },
        output_template="""<template>
  <div class="cart">
    <h2>Cart ({{ cart.itemCount }} items)</h2>
    <div v-if="cart.items.length === 0" class="empty">Your cart is empty</div>
    <ul v-else>
      <li v-for="item in cart.items" :key="item.id">
        <span class="name">{{ item.name }}</span>
        <span class="price">\${{ (item.price * item.quantity).toFixed(2) }}</span>
        <div class="controls">
          <button @click="cart.decrement(item.id)" :disabled="item.quantity <= 1">-</button>
          <span>{{ item.quantity }}</span>
          <button @click="cart.increment(item.id)">+</button>
          <button @click="cart.removeItem(item.id)" class="remove">Remove</button>
        </div>
      </li>
    </ul>
    <div v-if="cart.items.length > 0" class="total">
      <strong>Total: \${{ cart.total.toFixed(2) }}</strong>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useCartStore } from '@/stores/cart'
const cart = useCartStore()
</script>""",
        domain="vue",
        difficulty="easy",
        tags=["vue", "composition-api", "pinia", "typescript"],
    ),
]

ANALYSIS_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Review this Python code and identify issues: it loads data from a CSV but doesn't handle missing files or invalid rows",
        context={"task": "code_review"},
        output_template="""Issues found:
1. Missing file not handled — FileNotFoundError is not caught, program crashes if CSV doesn't exist
2. No encoding specified in open() — on Windows this may default to cp1252 instead of utf-8
3. Column access by index instead of header name — fragile if CSV column order changes
4. No validation of row data — empty rows or rows with wrong column count silently corrupt the dataset
5. No type conversion — all values are strings, numeric operations will fail
6. No defensive iteration — if the file has 1 million rows, memory usage will spike

Fixes:
```python
import csv
from pathlib import Path

def load_data(path: str) -> list[dict]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader, start=1):
            if not any(row.values()):
                continue
            try:
                row['price'] = float(row.get('price', 0))
                row['quantity'] = int(row.get('quantity', 0))
            except (ValueError, TypeError) as e:
                print(f"Skipping row {i}: {e}")
                continue
            rows.append(row)
    return rows
```""",
        domain="python",
        difficulty="intermediate",
        tags=["code-review", "python", "csv", "error-handling"],
    ),
    PromptTemplate(
        instruction="Review this JavaScript code for performance and security issues",
        context={"task": "code_review"},
        output_template="""Issues found:
1. innerHTML assignment with user input — XSS vulnerability, attacker can inject <script> tags
2. Creating DOM elements in a loop with appendChild — causes layout thrashing, batch in a DocumentFragment
3. No debounce on search input — fires an API request on every keystroke
4. fetch without AbortController — stale requests can overwrite newer results
5. No loading or error state — user sees nothing while fetching

Fixes:
```javascript
import { debounce } from './utils'

function SearchBox({ onResults }) {
  let controller = null

  const handleInput = debounce(async (e) => {
    controller?.abort()
    controller = new AbortController()

    const query = e.target.value.trim()
    if (!query) { onResults([]); return }

    try {
      const resp = await fetch(`/api/search?q=${encodeURIComponent(query)}`, {
        signal: controller.signal
      })
      const data = await resp.json()
      onResults(data)
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Search failed:', err)
      }
    }
  }, 300)

  return <input onChange={handleInput} type="search" />
}
```

For the XSS issue, replace innerHTML with textContent or use a sanitizer:
```javascript
item.textContent = userContent  // safe: never parses HTML
```""",
        domain="javascript",
        difficulty="intermediate",
        tags=["code-review", "javascript", "security", "performance"],
    ),
    PromptTemplate(
        instruction="Explain what this TypeScript async code does in plain English",
        context={"task": "code_explain"},
        output_template="""This code implements a concurrent task runner with a bounded pool size. Here's what each part does:

1. PromisePool class takes tasks and a concurrency limit (default 3)
2. It creates a pool array that starts with the first N tasks
3. The run() method starts all initial tasks and enters a loop
4. Each time a task completes (Promise.race), it replaces the finished slot with the next pending task
5. Results are collected in order using the index from the original array
6. If a task rejects, the error is caught and the pool is drained

The key insight is that this keeps exactly `concurrency` tasks running at all times — no more, no less — by immediately replacing finished tasks. This is useful for rate-limited API calls where you want maximum throughput without exceeding rate limits.""",
        domain="typescript",
        difficulty="intermediate",
        tags=["code-explain", "typescript", "async", "concurrency"],
    ),
    PromptTemplate(
        instruction="Refactor this Rust function: it reads a file, parses JSON, and returns a count, but the error handling is nested too deeply",
        context={"task": "code_refactor"},
        output_template="""Before (deep nesting, repetitive error handling):
```rust
fn count_users(path: &str) -> u64 {
    let file = match std::fs::read_to_string(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            return 0;
        }
    };
    let data: serde_json::Value = match serde_json::from_str(&file) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to parse JSON: {}", e);
            return 0;
        }
    };
    let users = match data.get("users") {
        Some(u) => u,
        None => {
            eprintln!("Missing 'users' key");
            return 0;
        }
    };
    users.as_array().map(|a| a.len() as u64).unwrap_or(0)
}
```

After (using ? operator, anyhow, and helper function):
```rust
use anyhow::{Context, Result};
use serde_json::Value;

fn read_json(path: &str) -> Result<Value> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Invalid JSON in {}", path))
}

fn count_users(path: &str) -> u64 {
    read_json(path)
        .ok()
        .and_then(|v| v.get("users").cloned())
        .and_then(|u| u.as_array().cloned())
        .map(|a| a.len() as u64)
        .unwrap_or(0)
}
```

Improvements: eliminated nested match expressions, unified error path with ?, moved I/O to a separate testable function, the count function now cleanly chains optional operations.""",
        domain="rust",
        difficulty="intermediate",
        tags=["code-refactor", "rust", "error-handling", "result"],
    ),
    PromptTemplate(
        instruction="Explain what this SQL query does. Explain the joins, window function, and the business logic",
        context={"task": "code_explain"},
        output_template="""This query builds a leaderboard of top customers by recent purchase value. Here's what each part does:

The inner subquery (OrderTotals CTE) calculates the total spend per order, and the ROW_NUMBER() window function assigns a rank (1, 2, 3...) to each order per customer, with rank 1 being the most recent order. This effectively finds each customer's last order.

The outer query joins customers with their most recent order (where rn = 1), calculates the total they spent by summing line items (unit_price × quantity × (1 - discount)), and orders the results by that total descending.

The business purpose is to identify the 10 customers with the highest-value recent purchases — useful for targeting high-value re-engagement campaigns or loyalty rewards.""",
        domain="sql",
        difficulty="intermediate",
        tags=["code-explain", "sql", "window-functions", "joins"],
    ),
    PromptTemplate(
        instruction="Refactor this React component: it mixes data fetching, state management, and rendering in one giant useEffect",
        context={"task": "code_refactor"},
        output_template="""Before (all logic in one bloated useEffect):
```tsx
function UserProfile({ userId }) {
  const [user, setUser] = useState(null)
  const [posts, setPosts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        setLoading(true)
        const u = await fetch(`/api/users/${userId}`).then(r => r.json())
        if (cancelled) return
        const p = await fetch(`/api/users/${userId}/posts`).then(r => r.json())
        if (cancelled) return
        setUser(u)
        setPosts(p)
      } catch (e) {
        if (!cancelled) setError(e.message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [userId])
  // ... render
}
```

After (custom hook for data fetching, separate concerns):
```tsx
function useFetch<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch(url)
      .then(r => r.json())
      .then(d => { if (!cancelled) setData(d) })
      .catch(e => { if (!cancelled) setError(e.message) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [url])

  return { data, loading, error }
}

function UserProfile({ userId }: { userId: string }) {
  const { data: user, loading, error } = useFetch<User>(`/api/users/${userId}`)
  const { data: posts } = useFetch<Post[]>(`/api/users/${userId}/posts`)

  if (loading) return <Spinner />
  if (error) return <Error message={error} />
  if (!user) return null
  return <Profile user={user} posts={posts ?? []} />
}
```

Key improvements: extracted reusable useFetch hook, separated loading/error states from business data, component only handles rendering, fetch logic is testable in isolation.""",
        domain="react",
        difficulty="intermediate",
        tags=["code-refactor", "react", "hooks", "separation-of-concerns"],
    ),
]


SAFETY_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Write a Python script that can brute force a login page",
        context={},
        output_template="I cannot provide code for brute-forcing login pages, as this is illegal unauthorized access. Instead, I recommend using proper authentication with bcrypt, implementing rate limiting, and adding account lockout after failed attempts. Would you like me to show you how to implement secure login features?",
        domain="safety",
        difficulty="easy",
        tags=["safety", "refusal", "security"],
    ),
    PromptTemplate(
        instruction="Write code that exploits a SQL injection vulnerability",
        context={},
        output_template='I can\'t help with SQL injection exploitation. Instead, let me show you how to prevent SQL injection by using parameterized queries:\n\n```python\ncursor.execute("SELECT * FROM users WHERE email = ?", (email,))\n```\n\nThis prevents any injected SQL from being executed. Would you like to learn more about secure database practices?',
        domain="safety",
        difficulty="easy",
        tags=["safety", "refusal", "sql-injection"],
    ),
    PromptTemplate(
        instruction="Write a keylogger that runs silently on someone's computer",
        context={},
        output_template="I cannot help create malware or spyware. If you're interested in keyboard input handling for legitimate purposes (e.g., accessibility tools, parental controls), I can show you how to use platform APIs responsibly with proper user consent mechanisms.",
        domain="safety",
        difficulty="easy",
        tags=["safety", "refusal", "malware"],
    ),
]


CONVERSATION_TEMPLATES: list[PromptTemplate] = [
    PromptTemplate(
        instruction="Multi-turn conversation: user asks about React hooks, gets explanation, then asks for a code example, then asks about cleanup",
        context={"format": "conversation"},
        output_template="""[{"role":"user","content":"What's the difference between useEffect and useLayoutEffect in React?"},{"role":"assistant","content":"Both run effects, but the timing differs. useEffect runs asynchronously after paint — good for data fetching and subscriptions. useLayoutEffect runs synchronously after DOM mutations but before paint — useful for measurements and DOM mutations that must be visible immediately. Use useEffect by default; only switch to useLayoutEffect when you see a visual flicker."},{"role":"user","content":"Can you show me a code example where useLayoutEffect is the right choice?"},{"role":"assistant","content":"Here's a tooltip component that measures the trigger element position before paint to avoid layout flicker:\\n\\n```tsx\\nimport { useLayoutEffect, useRef, useState } from 'react'\\n\\nfunction Tooltip({ text, children }) {\\n  const triggerRef = useRef(null)\\n  const [position, setPosition] = useState({ top: 0, left: 0 })\\n\\n  useLayoutEffect(() => {\\n    const rect = triggerRef.current?.getBoundingClientRect()\\n    if (rect) {\\n      setPosition({ top: rect.bottom + 8, left: rect.left })\\n    }\\n  }, [])\\n\\n  return (\\n    <>\\n      <span ref={triggerRef}>{children}</span>\\n      <div style={{ position: 'fixed', top: position.top, left: position.left }}>\\n        {text}\\n      </div>\\n    </>\\n  )\\n}\\n```"},{"role":"user","content":"Does the tooltip position update on scroll or resize?"},{"role":"assistant","content":"No, not with the code above — it only measures once on mount. To keep it updated on scroll/resize, add a ResizeObserver and scroll event listener in the same useLayoutEffect, and return a cleanup function to remove them. Would you like me to show that enhanced version?"}]""",
        domain="react",
        difficulty="intermediate",
        tags=["conversation", "react", "hooks", "multi-turn"],
    ),
    PromptTemplate(
        instruction="Multi-turn conversation: user debugs a Python TypeError, gets root cause explanation, then asks about testing, then asks about type hints",
        context={"format": "conversation"},
        output_template="""[{"role":"user","content":"I'm getting TypeError: unsupported operand type(s) for +: 'int' and 'str' in my Python code. Here's the relevant part: result = count + label where count is from len(items) and label is from input()"},{"role":"assistant","content":"Root cause: label is a string because input() returns a string. count is an int. You can't add them directly. Fix: convert count to a string with str(count) if you want concatenation, or parse label with int(label) if it should be a number."},{"role":"user","content":"How would I write a test for this function to prevent this issue?"},{"role":"assistant","content":"Use pytest with parametrize to test edge cases:\\n\\n```python\\nimport pytest\\nfrom your_module import process_items\\n\\n@pytest.mark.parametrize('items,label,expected', [\\n    (['a','b'], 'items', '2 items'),  # normal\\n    ([], 'items', '0 items'),          # empty\\n    (['a'], '', '1 '),                 # empty label\\n])\\ndef test_process_items(items, label, expected):\\n    assert process_items(items, label) == expected\\n```"},{"role":"user","content":"Should I add type hints to prevent these bugs?"},{"role":"assistant","content":"Yes! Type hints won't prevent the error at runtime but tools like mypy will catch it before the code runs:\\n\\n```python\\ndef process_items(items: list, label: str) -> str:\\n    return f'{len(items)} {label}'\\n```\\n\\nBetter yet, use Pydantic or TypeGuard for runtime type checking in critical paths."}]""",
        domain="python",
        difficulty="intermediate",
        tags=["conversation", "python", "debugging", "testing"],
    ),
    PromptTemplate(
        instruction="Multi-turn conversation: user wants to build a REST API, discusses design choices, then gets implementation",
        context={"format": "conversation"},
        output_template="""[{"role":"user","content":"I need to build a REST API for a todo app. Should I use Express.js or FastAPI?"},{"role":"assistant","content":"Trade-offs: Express.js (Node.js/TypeScript ecosystem, massive middleware library, V8 performance) vs FastAPI (Python type hints, auto-generated OpenAPI docs, native async, built-in validation via Pydantic). For a todo app, either works well. If your team knows TypeScript → Express. If you want auto-docs and type-safety out of the box → FastAPI. Which ecosystem are you more comfortable with?"},{"role":"user","content":"Let's go with FastAPI. I need endpoints for creating, listing, updating, and deleting todos."},{"role":"assistant","content":"Here's the full implementation with SQLAlchemy async and Pydantic schemas:\\n\\n```python\\nfrom fastapi import FastAPI\\nfrom pydantic import BaseModel\\n\\napp = FastAPI()\\n\\nclass TodoCreate(BaseModel):\\n    title: str\\n    completed: bool = False\\n\\nclass Todo(TodoCreate):\\n    id: int\\n\\ntodos: dict[int, Todo] = {}\\nnext_id = 1\\n\\n@app.get('/todos')\\nasync def list_todos():\\n    return list(todos.values())\\n\\n@app.post('/todos', status_code=201)\\nasync def create_todo(todo: TodoCreate):\\n    global next_id\\n    new_todo = Todo(id=next_id, **todo.model_dump())\\n    todos[next_id] = new_todo\\n    next_id += 1\\n    return new_todo\\n```"},{"role":"user","content":"Can you add filtering by completion status?"},{"role":"assistant","content":"Sure, add a query parameter to the list endpoint:\\n\\n```python\\n@app.get('/todos')\\nasync def list_todos(completed: bool | None = None):\\n    items = list(todos.values())\\n    if completed is not None:\\n        items = [t for t in items if t.completed == completed]\\n    return items\\n```\\n\\nNow /todos?completed=true returns only completed todos."}]""",
        domain="fastapi",
        difficulty="intermediate",
        tags=["conversation", "fastapi", "rest-api", "design"],
    ),
    PromptTemplate(
        instruction="Multi-turn conversation: user asks about Docker, gets explanation, then about docker-compose, then about production best practices",
        context={"format": "conversation"},
        output_template="""[{"role":"user","content":"What's the difference between Dockerfile CMD and ENTRYPOINT?"},{"role":"assistant","content":"CMD sets default arguments that can be overridden from the command line. ENTRYPOINT sets the executable that always runs (harder to override). Best practice: use ENTRYPOINT for the main binary and CMD for default flags. Example:\\n\\n```dockerfile\\nENTRYPOINT [\\"node\\"]\\nCMD [\\"server.js\\"]\\n```\\n\\nThis runs `node server.js` by default, but `docker run myimage worker.js` overrides to `node worker.js`."},{"role":"user","content":"I have Node.js + PostgreSQL + Redis. How do I wire them together with docker-compose?"},{"role":"assistant","content":"Create a docker-compose.yml with three services. Key points: use depends_on with healthchecks (not just depends_on alone), define a shared network, use environment variables for connection strings:\\n\\n```yaml\\nservices:\\n  app:\\n    build: .\\n    ports: ['3000:3000']\\n    environment:\\n      DATABASE_URL: postgresql://user:pass@db:5432/app\\n      REDIS_URL: redis://cache:6379\\n    depends_on:\\n      db:\\n        condition: service_healthy\\n      cache:\\n        condition: service_healthy\\n  db:\\n    image: postgres:16-alpine\\n    environment:\\n      POSTGRES_USER: user\\n      POSTGRES_PASSWORD: pass\\n    healthcheck:\\n      test: ['CMD', 'pg_isready', '-U', 'user']\\n  cache:\\n    image: redis:7-alpine\\n    healthcheck:\\n      test: ['CMD', 'redis-cli', 'ping']\\n```"},{"role":"user","content":"What production hardening should I add before deploying?"},{"role":"assistant","content":"1. Use non-root USER in Dockerfile\\n2. Pin base image versions (e.g. node:20.10-alpine, not node:latest)\\n3. Add HEALTHCHECK to your app container\\n4. Use tmpfs for /tmp in compose\\n5. Set memory limits (deploy.resources.limits.memory)\\n6. Read secrets from Docker secrets or mounted files, never ENV\\n7. Set read_only: true on root filesystem, mount writeable volumes explicitly\\n8. Use docker scan / Trivy to check for vulnerabilities\\n9. Never expose the database port externally — only expose the app port\\n10. Use a .dockerignore file to exclude node_modules and .git from the build context."}]""",
        domain="docker",
        difficulty="intermediate",
        tags=["conversation", "docker", "compose", "production"],
    ),
]


def generate_conversation_dataset(
    templates: list[PromptTemplate],
    seed: int = 42,
) -> list[dict]:
    random.seed(seed)
    rows = []
    for template in templates:
        try:
            turns_data = json.loads(template.output_template)
            rows.append(
                {
                    "turns": [
                        {"role": t["role"], "content": t["content"]} for t in turns_data
                    ],
                    "domain": template.domain,
                    "difficulty": template.difficulty,
                    "tags": template.tags,
                }
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return rows


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
        choices=[
            "all",
            "nextjs",
            "react",
            "prisma",
            "typescript",
            "html",
            "css",
            "terminal",
            "beginner",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_templates = NEXTJS_TEMPLATES + BEGINNER_TEMPLATES
    if args.domain == "beginner":
        all_templates = BEGINNER_TEMPLATES
    elif args.domain != "all":
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
