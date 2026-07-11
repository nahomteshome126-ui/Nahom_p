Fixing MongoDB connection errors and deploying to Vercel

If you see the error "Database connection failed. Please verify that MONGODB_URI is correct and that MongoDB Atlas allows access from all IPs (0.0.0.0/0)." it means your app cannot reach MongoDB. To fix this:

- Create a MongoDB Atlas cluster and a database user.
- Copy the connection string and replace `<username>`, `<password>`, and `<dbname>`.
- On your local machine, create a `.env` file using `.env.example` and set `MONGODB_URI`.
- In Atlas → Network Access, add your current IP or add a CIDR like `0.0.0.0/0` (less secure).
- Restart the server: `npm run server`.

Test locally:

```powershell
$env:MONGODB_URI="your-connection-string"
node test-mongo.js
```

Deploy to Vercel:

1. Push your branch to GitHub.
2. In the Vercel dashboard, import the GitHub repository.
3. In Project Settings → Environment Variables, add `MONGODB_URI` (and `GEMINI_API_KEY` if used) for the appropriate environment (Production/Preview/Development).
4. Trigger a redeploy.

Using Vercel CLI:

```bash
npm i -g vercel
vercel login
vercel link
vercel env add MONGODB_URI production
```

Security:

- Do not commit `.env` to source control.
- Prefer adding only specific IPs in Atlas instead of `0.0.0.0/0`.
